"""BESS day-ahead arbitrage backtest: what are the forecasts worth in EUR?

Plant: E = 100 MWh energy, P = 50 MW grid connection (2-hour battery), 85 % round-trip
efficiency split evenly between charge and discharge (eta = sqrt(0.85) per direction).

Protocol (no information that was not known at the decision time):
  * On the morning of D-1 (07:30 UTC information set) the hourly schedule for local day D is
    optimised by an LP on a *forecast* of day D and committed as price-taking volume bids in
    the day-ahead auction. Auction commitments are firm, so the schedule executes as planned
    and the state of charge follows the plan deterministically.
  * Settlement happens at the *actual* auction clearing prices of day D.
  * Forecasts are the stored out-of-sample online forecasts (strict rolling-origin,
    outputs/preds); the BOA combinations are rebuilt with weights that only ever saw past days.
    "Persistence" uses the D-1 price profile, which cleared in the D-2 auction and is known.

LP per day (profit maximisation), variables c_t (grid->battery, MWh) and d_t (battery MWh out):
  max sum_t p_t * (eta_d * d_t - c_t)
  s.t. SoC_k = sum_{t<k} (eta_c * c_t - d_t) in [0, E] for all k,  SoC_24 = 0 (daily cycle:
       local midnight lies in the overnight trough, so this costs almost nothing and keeps the
       strategies exactly comparable), 0 <= c_t <= P, 0 <= eta_d * d_t <= P (grid-side limit).

Usage: python src/bess_backtest.py
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

import evaluate as ev
from models import BOACombiner

E, P_GRID = 100.0, 50.0                 # MWh, MW
ETA = np.sqrt(0.85)                     # per-direction efficiency (85 % round trip)
PERIODS = [("2025", "2025-01-01", "2025-12-31"), ("2026 (Jan-Sep)", "2026-01-01", "2026-09-27")]


def day_schedule(q):
    """Optimal (c, d) for one day given expected prices q (24,). Returns grid-side energies."""
    # variables x = [c_0..c_23, d_0..d_23]
    cost = np.r_[q, -ETA * q]                               # minimise -profit
    L = np.tril(np.ones((24, 24)))                          # cumulative sums: SoC_k for k = 1..24
    A_soc = np.hstack([ETA * L, -L])                        # SoC after hour k
    A_ub = np.vstack([A_soc, -A_soc])
    b_ub = np.r_[np.full(24, E), np.zeros(24)]
    A_eq, b_eq = A_soc[-1:], np.zeros(1)                    # SoC_24 = 0
    bounds = [(0, P_GRID)] * 24 + [(0, P_GRID / ETA)] * 24
    r = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    assert r.status == 0, r.message
    return r.x[:24], r.x[24:]


def backtest(forecast, actual):
    """Daily LP on `forecast`, settled at `actual`. Returns per-day profit, throughput, SoC check."""
    profit, thru = np.zeros(len(actual)), np.zeros(len(actual))
    for i in range(len(actual)):
        c, d = day_schedule(forecast[i])
        soc = np.cumsum(ETA * c - d)                        # bookkeeping: battery-side SoC path
        assert soc.min() > -1e-6 and soc.max() < E + 1e-6 and abs(soc[-1]) < 1e-6
        profit[i] = np.dot(actual[i], ETA * d - c)          # EUR: sell eta_d*d, buy c
        thru[i] = d.sum()                                   # battery-side discharged energy
    return profit, thru


def main():
    days, y, P = ev.load()
    # OOS combination forecasts (combiner state only ever uses past days)
    for key, members in [("hyb_boa_all", ev.HYBRID_ALL), ("hyb_boa_sel", ev.HYBRID_SEL)]:
        P[key], _ = ev.run_combiner(BOACombiner(len(members)), members, P, y)
    pers = np.vstack([np.full((1, 24), np.nan), y[:-1]])    # D-1 profile, known at the decision time
    strategies = [("Perfect Foresight (Obergrenze)", y), ("Hybrid BOA alle 13", P["hyb_boa_all"]),
                  ("BOA selected 6", P["hyb_boa_sel"]), ("LEAR", P["lear"]),
                  ("Persistenz (D-1-Profil)", pers)]

    for pname, lo, hi in PERIODS:
        mask = (days >= lo) & (days <= hi)
        if pers is not None:
            mask &= np.isfinite(pers).all(1)
        n = mask.sum()
        rows, pf = [], None
        for name, f in strategies:
            profit, thru = backtest(f[mask], y[mask])
            pf = profit.sum() if pf is None else pf
            rows.append({"Strategie": name, "Gewinn (EUR)": profit.sum(),
                         "EUR/kW/Jahr": profit.sum() / (P_GRID * 1e3) / (n / 365.25),
                         "% von PF": 100 * profit.sum() / pf,
                         "Zyklen/Tag": thru.mean() / E,
                         "Verlusttage": int((profit < 0).sum())})
        res = pd.DataFrame(rows).set_index("Strategie")
        with pd.option_context("display.float_format", "{:,.1f}".format, "display.width", 140):
            print(f"\n=== {pname}: {n} Tage, 100 MWh / 50 MW, Round-Trip 85 %, taeglicher LP ===")
            print(res.to_string())
    print("\nHinweis: Gewinn = Arbitrage-Rohertrag der DA-Auktion (ohne Netzentgelte/Degradation);"
          "\n'Verlusttage' = Tage, an denen der auf der Prognose gebaute Fahrplan real Geld verlor.")


if __name__ == "__main__":
    main()
