"""QRA-15 experiment: quantile regression averaging over the 13 point members PLUS the two
dedicated GBM tail members (pinball-loss gbm_q10 / gbm_q90, EXTRA_FEATS=2, weekly refit).
Idea: QRA's regressors are all conditional-mean forecasts, which systematically under-spread
in spike regimes; native quantile forecasts give the QRA design matrix direct information
about the conditional tails.

Same protocol as qra_x_experiment: raw QRA refit daily (182-day window), then + adaptive
conformal; overall pinball on 2024 (selection) / 2025-26 (holdout) plus spike-hour metrics,
DM test on the holdout.

Usage: python src/qra15_experiment.py [--workers N]
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import evaluate as ev
from probabilistic import MIN_DAYS, QUANTILES, conformalize, pinball, qra
from qra_x_experiment import report, spike_mask

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "qra15_raw.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    days, y, P = ev.load()
    F = np.stack([P[m] for m in ev.HYBRID_ALL], -1)
    Fq = np.stack([P[m] for m in ev.HYBRID_ALL + ["gbm_q10", "gbm_q90"]], -1)

    if OUT.exists():
        Qq = np.load(OUT)
    else:
        print(f"fitting QRA-15 ({Fq.shape[-1]} regressors, {len(days)} days, refit daily) ...", flush=True)
        Qq = qra(Fq, y, MIN_DAYS, 1, args.workers)
        np.save(OUT, Qq)
    from probabilistic import cached_qra
    class A: refit_every, workers = 1, args.workers
    Q = cached_qra(F, y, A())                                  # plain QRA (cache hit)

    spike = spike_mask(days, y)
    print("\nRaw QRA:")
    report("QRA alle 13", Q, y, days, spike)
    report("QRA-15 (+gbm_q10/q90)", Qq, y, days, spike)
    print("+ adaptive conformal:")
    Qc = conformalize(Q, y, 2 * MIN_DAYS, gamma=0.005)
    Qqc = conformalize(Qq, y, 2 * MIN_DAYS, gamma=0.005)
    report("QRA + ACI", Qc, y, days, spike)
    report("QRA-15 + ACI", Qqc, y, days, spike)

    hold = (days >= "2025-01-01") & np.isfinite(Qc).all((1, 2)) & np.isfinite(Qqc).all((1, 2))
    from scipy import stats
    da = np.mean([pinball(y, Qc[..., i], t) for i, t in enumerate(QUANTILES)], 0).sum(1)
    db = np.mean([pinball(y, Qqc[..., i], t) for i, t in enumerate(QUANTILES)], 0).sum(1)
    d = (da - db)[hold]
    print(f"\nDM p(QRA-15+ACI besser, Holdout) = {1 - stats.norm.cdf(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))):.4f}")


if __name__ == "__main__":
    main()
