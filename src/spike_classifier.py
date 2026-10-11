"""Online spike classifier + quantile widening experiment.

Stage 1: every day D (decision time of D-1) a gradient-boosting classifier estimates the
probability that D contains at least one spike hour (price above the causal 97.5 % quantile
of the trailing 365 days). Features are tightness variables known at the decision time
(residual load, wind, demand forecast, fuel cost, D-1 price level, calendar). Weekly refits
on all past labelled days - strict rolling origin, as everywhere in this project.

Stage 2: use the spike probability to widen the upper quantiles of the production forecast
(QRA all members + ACI):  q_tau' = q_tau + alpha * p_spike * (q_tau - q50)  for tau > 0.5.
alpha is selected on 2024 only and validated on the untouched 2025-26 holdout.

Usage: python src/spike_classifier.py
"""
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd

import evaluate as ev
from data import DayAheadData
from probabilistic import MIN_DAYS, QUANTILES, cached_qra, conformalize, pinball

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["res_load_max", "res_load_mean", "wind_min", "wind_mean", "demand_max",
         "fuel_cost", "p_d1_max", "p_d1_mean", "temp_mean", "is_weekend", "doy_sin", "doy_cos",
         "surplus", "surplus_rel"]                     # NESO OC2 margin (issue of D-2, OCNMFD)


def margin_features(days):
    m = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "raw" / "margin_forecast_ocnmfd.csv",
                    parse_dates=["delivery_day"])
    s = m.set_index("delivery_day")["surplus"].reindex(days)
    return s.values / 1e3                              # GW; NaN before mid-2021 (HistGB handles NaN)


def day_features(data, ix):
    c = data.tab_cols.index
    T = data.tab[ix].astype(np.float64)
    X = np.column_stack([T[:, :, c("res_load")].max(1), T[:, :, c("res_load")].mean(1),
                         T[:, :, c("wind_da")].min(1), T[:, :, c("wind_da")].mean(1),
                         T[:, :, c("demand_fc")].max(1), T[:, 0, c("fuel_cost")],
                         T[:, 0, c("p_d1_max")], T[:, 0, c("p_d1_mean")],
                         T[:, :, c("temperature")].mean(1), T[:, 0, c("is_weekend")],
                         T[:, 0, c("doy_sin")], T[:, 0, c("doy_cos")]])
    return X


def main():
    days, y, P = ev.load()
    data = DayAheadData()
    ix = [data.index_of(d) for d in days]
    X = day_features(data, ix)
    sur = margin_features(days)
    X = np.column_stack([X, sur, sur * 1e3 / np.maximum(X[:, 4], 1.0)])   # GW and share of peak demand

    flat = pd.Series(y.ravel(), index=np.repeat(days, 24))
    thr = flat.rolling(24 * 365).quantile(0.975).shift(1).values.reshape(-1, 24)
    spike_h = y > thr                                   # hourly spikes (causal threshold)
    label = spike_h.any(1).astype(int)                  # spike day
    ok = np.isfinite(thr).all(1)                        # threshold needs a year of history

    from sklearn.ensemble import HistGradientBoostingClassifier
    prob = np.full(len(days), np.nan)
    model = None
    start = int(np.argmax(ok)) + 56
    for t in range(start, len(days)):
        if model is None or (t - start) % 7 == 0:
            tr = np.arange(np.argmax(ok), t)            # all labelled past days
            model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                   min_samples_leaf=30, l2_regularization=1.0,
                                                   random_state=0).fit(X[tr], label[tr])
        prob[t] = model.predict_proba(X[t:t + 1])[0, 1]
    np.save(ROOT / "outputs" / "spike_prob.npy", prob)

    from sklearn.metrics import roc_auc_score, brier_score_loss
    for lab, m in [("2024", (days >= "2024-01-01") & (days <= "2024-12-31")), ("2025-26", days >= "2025-01-01")]:
        mm = m & np.isfinite(prob)
        print(f"Klassifikator {lab}: AUC {roc_auc_score(label[mm], prob[mm]):.3f}  "
              f"Brier {brier_score_loss(label[mm], prob[mm]):.3f}  Basisrate {label[mm].mean():.1%}")

    # ---------------- stage 2: widen the upper quantiles of QRA+ACI ----------------
    F = np.stack([P[m] for m in ev.HYBRID_ALL], -1)
    class A: refit_every, workers = 1, 6
    Qc = conformalize(cached_qra(F, y, A()), y, 2 * MIN_DAYS, gamma=0.005)
    i50 = list(QUANTILES).index(0.50)
    up = QUANTILES > 0.5
    spike_hold = spike_h & (days >= "2025-01-01")[:, None]

    def widened(alpha):
        Qw = Qc.copy()
        w = np.where(np.isfinite(prob), prob, 0.0)[:, None, None] * alpha
        Qw[..., up] = Qc[..., up] + w * np.maximum(Qc[..., up] - Qc[..., [i50]], 0)
        return Qw

    def apl(Q, m, sp=None):
        mm = m & np.isfinite(Q).all((1, 2)) & np.isfinite(prob)
        r = np.stack([pinball(y[mm], Q[mm][..., i], t) for i, t in enumerate(QUANTILES)], -1)
        if sp is not None:
            return r[sp[mm]].mean()
        return r.mean()

    sel = (days >= "2024-01-01") & (days <= "2024-12-31")
    hold = days >= "2025-01-01"
    print("\nalpha  APL 2024   APL Holdout  Spike-APL Hold  P(y<=q90|Spike,Hold)")
    i90 = list(QUANTILES).index(0.90)
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5]:
        Qw = widened(alpha)
        mm = hold & np.isfinite(Qw).all((1, 2)) & np.isfinite(prob)
        cov = (y[mm] <= Qw[mm][..., i90])[spike_h[mm]].mean()
        print(f"{alpha:4.1f}  {apl(Qw, sel):8.3f}  {apl(Qw, hold):10.3f}  {apl(Qw, hold, spike_h):13.2f}  {cov:18.1%}")


if __name__ == "__main__":
    main()
