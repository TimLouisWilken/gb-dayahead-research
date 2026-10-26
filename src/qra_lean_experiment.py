"""Lean-QRA and centring experiment.

Two questions about the quantile product (QRA-15 + adaptive conformal):
  1. Fewer regressors: QRA over the 6 BOA-selected members (one per model family) plus the two
     GBM tail members (8 regressors, 9 coefficients per formula) instead of all 13 + 2.
     Correlated siblings make unconstrained QRA coefficients unstable on a 182-day window.
  2. Centring: the BOA-selected point forecast has a clearly lower MAE than QRA's q50
     (8.56 vs 9.25). Shift all 19 raw QRA quantiles by (BOA - q50), so the band keeps
     QRA's shape but is centred on the better median, then recalibrate with adaptive conformal.

Variants (pre-registered): A = QRA-15, B = QRA-8, A_c / B_c = centred versions, all + ACI.
Selection on 2024, one-sided DM test on the 2025-26 holdout vs A.

Usage: python src/qra_lean_experiment.py [--workers N]
"""
import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import stats

import evaluate as ev
from models import BOACombiner
from probabilistic import I50, MIN_DAYS, QRA_TAILS, QUANTILES, cached_qra, centre, conformalize, pinball
from qra_x_experiment import spike_mask

def summary(Q, y, m, spike):
    Qm, ym = Q[m], y[m]
    row = {"APL": np.mean([pinball(ym, Qm[..., i], t).mean() for i, t in enumerate(QUANTILES)]),
           "MAE q50": np.abs(ym - Qm[..., I50]).mean()}
    for cov in (50, 80, 90):
        lo = list(QUANTILES).index(round((1 - cov / 100) / 2, 2))
        hi = list(QUANTILES).index(round(1 - (1 - cov / 100) / 2, 2))
        row[f"cov{cov}"] = ((ym >= Qm[..., lo]) & (ym <= Qm[..., hi])).mean() * 100
        row[f"w{cov}"] = (Qm[..., hi] - Qm[..., lo]).mean()
    sp = spike[m]
    row["Spike-APL"] = np.mean([pinball(ym[sp], Qm[..., i][sp], t).mean() for i, t in enumerate(QUANTILES)])
    return row


def daily_apl(Q, y):
    return np.mean([pinball(y, Q[..., i], t) for i, t in enumerate(QUANTILES)], 0).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    days, y, P = ev.load()
    point, _ = ev.run_combiner(BOACombiner(len(ev.HYBRID_SEL)), ev.HYBRID_SEL, P, y)   # served weights

    class A: refit_every, workers = 1, args.workers
    raw = {}
    for key, members in [("A", ev.HYBRID_ALL + QRA_TAILS), ("B", ev.HYBRID_SEL + QRA_TAILS)]:
        print(f"QRA {key}: {len(members)} regressors (cached if available) ...", flush=True)
        raw[key] = cached_qra(np.stack([P[m] for m in members], -1), y, A())
    raw["A_c"], raw["B_c"] = centre(raw["A"], point), centre(raw["B"], point)
    Q = {k: conformalize(v, y, 2 * MIN_DAYS, gamma=0.005) for k, v in raw.items()}

    label = {"A": "QRA-15 + ACI (baseline)", "B": "QRA-8 (sel6 + tails) + ACI",
             "A_c": "QRA-15 centred + ACI", "B_c": "QRA-8 centred + ACI"}
    ok = np.all([np.isfinite(q).all((1, 2)) for q in Q.values()], 0)
    spike = spike_mask(days, y)
    periods = {"2024 (selection)": ok & (days >= "2024-01-01") & (days <= "2024-12-31"),
               "2025-26 (holdout)": ok & (days >= "2025-01-01"),
               "2024-26 (all)": ok & (days >= "2024-01-01")}
    rows = []
    for per, m in periods.items():
        for k in Q:
            rows.append({"period": per, "variant": label[k], **summary(Q[k], y, m, spike)})
    res = pd.DataFrame(rows).set_index(["period", "variant"])
    res.to_csv(ev.OUT / "qra_lean_metrics.csv")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
        print(res.to_string())
    print(f"\nBOA-selected point forecast MAE 2024-26: {np.abs(y - point)[periods['2024-26 (all)']].mean():.3f}")

    hold = periods["2025-26 (holdout)"]
    base = daily_apl(Q["A"], y)
    for k in ["B", "A_c", "B_c"]:
        d = (base - daily_apl(Q[k], y))[hold]
        p = 1 - stats.norm.cdf(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
        print(f"DM p({label[k]} better than baseline, holdout) = {p:.4f}")
    d = (daily_apl(Q["B"], y) - daily_apl(Q["B_c"], y))[hold]
    print(f"DM p(QRA-8 centred better than QRA-8, holdout) = "
          f"{1 - stats.norm.cdf(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))):.4f}")


if __name__ == "__main__":
    main()
