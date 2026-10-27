"""Probabilistic day-ahead forecasts on top of the online point forecasts.

Methods (both use only out-of-sample forecasts of days < t, recalibrated on a rolling window):

  HS    historical simulation: Hybrid-BOA point forecast + empirical quantiles of its own
        errors over the last WINDOW days, separately for each hour
  QRA   Quantile Regression Averaging (Nowotarski & Weron, 2015): for every hour and quantile a
        linear quantile regression of the price on the member point forecasts
        (LSTM, LSTM+RF, RF long, RF short, LEAR)

  QRA+ACI as QRA+C, but the residual quantile level adapts online after every day
        (adaptive conformal inference, Gibbs & Candès, 2021; gamma = 0.005 from the paper)
  QRA+C QRA with a conformal correction (cf. conformalized quantile regression, Romano, Patterson
        & Candès, 2019): each quantile of each hour is shifted by the empirical tau-quantile of
        its own past out-of-sample residuals y - q (last WINDOW days), restoring calibration
  QRA-8 centred (exported product): QRA over the 6 BOA-selected members + the 2 GBM tails, all
        19 raw quantiles shifted by (BOA-selected point forecast - q50), then + ACI. Fewer,
        less correlated regressors stabilise the coefficients; centring puts the band on the
        better median and makes q50 consistent with the published point forecast

Quantile crossing is removed by sorting (Chernozhukov, Fernández-Val & Galichon, 2010).
Evaluation: average pinball loss over 19 quantiles (≈ CRPS/2, Gneiting & Raftery, 2007),
empirical coverage and width of central 50/80/90 % intervals, Kupiec (1995) coverage test.

Usage: python src/probabilistic.py [--refit-every 1] [--pred-dir outputs/preds]
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import QuantileRegressor

import evaluate as ev
from models import BOACombiner

ROOT = Path(__file__).resolve().parents[1]
QUANTILES = np.round(np.arange(0.05, 0.96, 0.05), 2)            # 19 quantiles
QRA_MEMBERS = ["lstm", "lstm_rf", "rf_long", "rf_short", "lear"]
QRA_MEMBERS_NN_RF = ["lstm", "lstm_rf", "rf_long", "rf_short"]       # pure LSTM + RF variant (no LEAR)
# dedicated GBM tail forecasts (pinball loss q10/q90, EXTRA_FEATS=2) as extra QRA regressors:
# mean forecasts under-spread in spike regimes, the tails inform the design matrix directly.
# Selected on 2024 (spike coverage 65->68 %), confirmed on the 2025-26 holdout (DM p = 0.038).
QRA_TAILS = ["gbm_q10", "gbm_q90"]
# exported product: QRA over the 6 BOA-selected members + tails, centred on the BOA-selected point
# forecast. Selected on 2024 (APL 3.096 vs 3.262 for QRA-15), confirmed on the 2025-26 holdout
# (3.554 vs 3.780, DM p < 0.001; centring alone vs uncentred QRA-8: p = 0.002). src/qra_lean_experiment.py
QRA_SEL = ["lstm_rf", "gbm", "transformer", "lear", "merit", "chronos"] + QRA_TAILS
WINDOW = 182                                                     # calibration window (days)
MIN_DAYS = 56                                                    # first forecast needs this many past days
I50 = list(QUANTILES).index(0.5)


def centre(Q, point):
    """Shift all quantiles by (point - q50): keeps QRA's band shape, uses the point forecast as median."""
    return np.sort(Q + (point - Q[..., I50])[..., None], axis=-1)


def pinball(y, q, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


# ------------------------------------------------------------- QRA (parallel over days)
def _qra_block(args):
    F, y, days_idx, refit_every = args          # F: (n, 24, K) member forecasts, y: (n, 24)
    out = {}
    models = None
    for k, t in enumerate(days_idx):
        lo = max(0, t - WINDOW)
        if models is None or k % refit_every == 0:
            models = {}
            for h in range(24):
                X, Y = F[lo:t, h, :], y[lo:t, h]
                ok = np.isfinite(Y) & np.isfinite(X).all(1)       # live: D-1 may lack its last hour
                X, Y = X[ok], Y[ok]
                models[h] = [QuantileRegressor(quantile=q, alpha=0, solver="highs").fit(X, Y) for q in QUANTILES]
        out[t] = np.stack([[m.predict(F[t:t + 1, h, :])[0] for m in models[h]] for h in range(24)])
    return out


def qra(F, y, start, refit_every, workers):
    days_idx = np.arange(start, len(y))
    blocks = np.array_split(days_idx, workers)
    res = {}
    with ProcessPoolExecutor(workers) as ex:
        for part in ex.map(_qra_block, [(F, y, b, refit_every) for b in blocks if len(b)]):
            res.update(part)
    Q = np.full((len(y), 24, len(QUANTILES)), np.nan)
    for t, q in res.items():
        Q[t] = q
    return np.sort(Q, axis=-1)                   # rearrangement: no quantile crossing


def conformalize(Q, y, start, gamma=0.0):
    """Shift every quantile by an empirical quantile of its past OOS residuals (days < t only).

    gamma > 0 adds adaptive conformal inference (Gibbs & Candès, 2021): the residual quantile level
    of each (hour, quantile) is nudged after every day, a <- a + gamma * (tau - 1[y <= q]), so
    coverage recovers after volatility shifts. gamma = 0.005 is the value used in the paper."""
    Qc = np.full_like(Q, np.nan)
    level = np.tile(QUANTILES, (24, 1)).astype(float)                         # (24, nq)
    for t in range(start, len(y)):
        past = np.arange(max(0, t - WINDOW), t)
        past = past[np.isfinite(Q[past]).all((1, 2)) & np.isfinite(y[past]).all(1)]
        if len(past) < MIN_DAYS:
            continue
        R = y[past][:, :, None] - Q[past]                                     # (w, 24, nq)
        lv = np.clip(level, 0.001, 0.999)
        shift = np.array([[np.quantile(R[:, h, i], lv[h, i]) for i in range(len(QUANTILES))] for h in range(24)])
        Qc[t] = np.sort(Q[t] + shift, axis=-1)
        if gamma and np.isfinite(y[t]).all():                                 # y[t] is known only after day t
            level += gamma * (QUANTILES[None, :] - (y[t][:, None] <= Qc[t]))
    return Qc


def cached_qra(F, y, args):
    """QRA is the expensive step; cache it by its inputs so re-evaluation is instant."""
    import hashlib
    key = hashlib.md5(F.tobytes() + y.tobytes() + str(args.refit_every).encode()).hexdigest()[:12]
    path = ev.OUT / f"qra_cache_{key}.npy"
    if path.exists():
        return np.load(path)
    Q = qra(F, y, MIN_DAYS, args.refit_every, args.workers)
    np.save(path, Q)
    return Q


def historical_simulation(point, y, start):
    Q = np.full((len(y), 24, len(QUANTILES)), np.nan)
    for t in range(start, len(y)):
        err = y[max(0, t - WINDOW):t] - point[max(0, t - WINDOW):t]         # (w, 24)
        Q[t] = point[t][:, None] + np.quantile(err, QUANTILES, axis=0).T
    return Q


# ------------------------------------------------------------- evaluation
def kupiec(hits, p):
    """Unconditional coverage LR test: H0 'hit rate = p'. Returns the p-value."""
    n, x = hits.size, hits.sum()
    pi = x / n
    if pi in (0, 1):
        return 0.0
    lr = -2 * (x * np.log(p) + (n - x) * np.log(1 - p) - x * np.log(pi) - (n - x) * np.log(1 - pi))
    return 1 - stats.chi2.cdf(lr, 1)


def evaluate_quantiles(name, Q, y, mask):
    Qm, ym = Q[mask], y[mask]
    row = {"method": name,
           "APL (avg pinball)": np.mean([pinball(ym, Qm[..., i], tau).mean() for i, tau in enumerate(QUANTILES)])}
    for cov in (50, 80, 90):
        lo_i = int(np.where(np.isclose(QUANTILES, (1 - cov / 100) / 2))[0][0])
        hi_i = int(np.where(np.isclose(QUANTILES, 1 - (1 - cov / 100) / 2))[0][0])
        hit = (ym >= Qm[..., lo_i]) & (ym <= Qm[..., hi_i])
        row[f"cov {cov}%"] = hit.mean() * 100
        row[f"width {cov}%"] = (Qm[..., hi_i] - Qm[..., lo_i]).mean()
        row[f"Kupiec p {cov}%"] = kupiec(hit.ravel(), cov / 100)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-every", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--pred-dir", default=None)
    args = ap.parse_args()
    if args.pred_dir:
        ev.PRED = Path(args.pred_dir).resolve()

    days, y, P = ev.load()
    hyb = [m for m in ev.HYBRID if m in P]
    P["hyb_boa"], _ = ev.run_combiner(BOACombiner(len(hyb)), hyb, P, y)
    F = np.stack([P[m] for m in QRA_MEMBERS], -1)
    F_nr = np.stack([P[m] for m in QRA_MEMBERS_NN_RF], -1)

    Q = {"HS (Hybrid BOA)": historical_simulation(P["hyb_boa"], y, MIN_DAYS),
         "QRA (members)": cached_qra(F, y, args)}
    Q["QRA + conformal"] = conformalize(Q["QRA (members)"], y, 2 * MIN_DAYS)
    Q["QRA + adaptive conformal"] = conformalize(Q["QRA (members)"], y, 2 * MIN_DAYS, gamma=0.005)
    # members that forecast distributions themselves (no post-processing)
    for name, key in [("Distributional LSTM (direct)", "ddnn"), ("Chronos-Bolt (direct)", "chronos")]:
        f = ev.PRED / f"{key}.npz"
        if f.exists():
            z = np.load(f)
            qd = pd.DataFrame(z["quant"].reshape(len(z["days"]), -1), index=pd.DatetimeIndex(z["days"]))
            Q[name] = np.sort(qd.reindex(days).values.reshape(len(days), 24, -1), axis=-1)
    if all(m in P for m in ev.HYBRID_ALL):
        Q["QRA all models"] = cached_qra(np.stack([P[m] for m in ev.HYBRID_ALL], -1), y, args)
        Q["QRA all models + adaptive conformal"] = conformalize(Q["QRA all models"], y, 2 * MIN_DAYS, gamma=0.005)
    if all(m in P for m in ev.HYBRID_ALL + QRA_TAILS):
        Q["QRA-15 (+GBM tails)"] = cached_qra(np.stack([P[m] for m in ev.HYBRID_ALL + QRA_TAILS], -1), y, args)
        Q["QRA-15 + adaptive conformal"] = conformalize(Q["QRA-15 (+GBM tails)"], y, 2 * MIN_DAYS, gamma=0.005)
    if all(m in P for m in QRA_SEL):
        P["hyb_boa_sel"], _ = ev.run_combiner(BOACombiner(len(ev.HYBRID_SEL)), ev.HYBRID_SEL, P, y)
        Q["QRA-8 (sel6 + tails)"] = cached_qra(np.stack([P[m] for m in QRA_SEL], -1), y, args)
        Q["QRA-8 centred + adaptive conformal"] = conformalize(centre(Q["QRA-8 (sel6 + tails)"], P["hyb_boa_sel"]),
                                                               y, 2 * MIN_DAYS, gamma=0.005)
    Q["QRA LSTM+RF only"] = cached_qra(F_nr, y, args)
    Q["QRA LSTM+RF only + adaptive conformal"] = conformalize(Q["QRA LSTM+RF only"], y, 2 * MIN_DAYS, gamma=0.005)

    mask = days >= pd.Timestamp(ev.EVAL_START)
    res = pd.DataFrame([evaluate_quantiles(k, v, y, mask) for k, v in Q.items()]).set_index("method")
    res.to_csv(ev.OUT / "prob_metrics.csv")
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        print(f"Probabilistic evaluation {ev.EVAL_START} -> {days[-1].date()} ({mask.sum()} days):")
        print(res.to_string())

    # one-sided DM test on the daily summed pinball loss: QRA better than HS?
    def daily_apl(Qx):
        return np.mean([pinball(y, Qx[..., i], tau) for i, tau in enumerate(QUANTILES)], 0).sum(1)
    for a, b in [("HS (Hybrid BOA)", "QRA (members)"), ("HS (Hybrid BOA)", "QRA + adaptive conformal"),
                 ("QRA (members)", "QRA + conformal"), ("QRA (members)", "QRA + adaptive conformal"),
                 ("HS (Hybrid BOA)", "QRA LSTM+RF only + adaptive conformal"),
                 ("QRA LSTM+RF only + adaptive conformal", "QRA + adaptive conformal"),
                 ("QRA + adaptive conformal", "QRA all models + adaptive conformal"),
                 ("QRA all models + adaptive conformal", "QRA-15 + adaptive conformal"),
                 ("QRA-15 + adaptive conformal", "QRA-8 centred + adaptive conformal"),
                 ("Distributional LSTM (direct)", "QRA + adaptive conformal")]:
        if a not in Q or b not in Q:
            continue
        d = (daily_apl(Q[a]) - daily_apl(Q[b]))[mask]
        print(f"DM test, {b} better than {a}: p = {1 - stats.norm.cdf(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))):.4f}")
    print("Note: Kupiec assumes independent hits; hourly hits are autocorrelated and n is ~24k, so it rejects\n"
          "even small deviations - read the coverage levels themselves.")

    # export quantiles of the best method: QRA-8 centred on the BOA-selected point forecast,
    # + adaptive conformal (see QRA_SEL); falls back gracefully.
    best = next(k for k in ["QRA-8 centred + adaptive conformal", "QRA-15 + adaptive conformal",
                            "QRA all models + adaptive conformal",
                            "QRA + adaptive conformal"] if k in Q)
    hours = (days.values[:, None] + np.arange(24) * np.timedelta64(1, "h")).ravel()
    out = pd.DataFrame(Q[best].reshape(-1, len(QUANTILES)), columns=[f"q{int(q * 100):02d}" for q in QUANTILES],
                       index=pd.DatetimeIndex(hours, name=f"datetime_{ {'GB': 'uk'}.get(ev.MARKET, ev.MARKET.lower())}_local"))
    out["actual"] = y.ravel()
    out[mask.repeat(24)].to_csv(ev.OUT / "forecasts_quantiles.csv")
    print(f"quantiles of '{best}' -> outputs/forecasts_quantiles.csv")

    figures(days, y, Q, mask, best)


def figures(days, y, Q, mask, best):
    C, INK, INK2 = ev.C, ev.INK, ev.INK2
    # reliability: empirical vs nominal quantile level
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="perfect calibration")
    show = ["HS (Hybrid BOA)", "QRA LSTM+RF only + adaptive conformal", "QRA + adaptive conformal",
            "QRA all models + adaptive conformal", "Distributional LSTM (direct)", "Chronos-Bolt (direct)"]
    for c, k in zip(C, [k for k in show if k in Q]):
        Qx = Q[k]
        emp = [(y[mask] <= Qx[mask][..., i]).mean() for i in range(len(QUANTILES))]
        ax.plot(QUANTILES, emp, color=c, marker="o", ms=4, label=k)
    ax.set_xlabel("nominal quantile level"); ax.set_ylabel("share of prices below the quantile")
    ax.set_title("Reliability of the quantile forecasts", loc="left", color=INK)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(ev.FIG / "prob_reliability.png", dpi=150); plt.close(fig)

    # fan chart, two weeks
    sel = (days >= "2025-01-13") & (days < "2025-01-27")
    if sel.sum() == 0:
        return
    t = pd.date_range(days[sel][0], periods=sel.sum() * 24, freq="h")
    Qs = Q[best][sel].reshape(-1, len(QUANTILES))
    fig, ax = plt.subplots(figsize=(11, 4))
    for (lo, hi), a in [((0.05, 0.95), 0.18), ((0.10, 0.90), 0.28), ((0.25, 0.75), 0.45)]:
        i, j = int(np.where(np.isclose(QUANTILES, lo))[0][0]), int(np.where(np.isclose(QUANTILES, hi))[0][0])
        ax.fill_between(t, Qs[:, i], Qs[:, j], color=C[0], alpha=a, lw=0, label=f"{round((hi - lo) * 100)} % interval")
    ax.plot(t, Qs[:, int(np.where(np.isclose(QUANTILES, 0.5))[0][0])], color=C[0], lw=1.4, label="median")
    ax.plot(t, y[sel].ravel(), color=INK, lw=1.3, label="actual")
    ax.set_ylabel("EUR/MWh"); ax.legend(ncol=5, fontsize=9, loc="upper left")
    ax.set_title(f"Probabilistic day-ahead forecast ({best}), January 2025", loc="left", color=INK)
    fig.tight_layout(); fig.savefig(ev.FIG / "prob_fan_chart.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
