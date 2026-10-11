"""QRA-X experiment: quantile regression averaging over all members PLUS fundamental
regressors of the delivery day (residual-load tightness and CCGT fuel cost, both known at the
decision time) - cf. Maciejowska & Nowotarski and Uniejewski & Weron on QRA variants.

Computes the raw QRA-X quantiles over the whole online period (refit daily, 182-day window),
stores them next to the plain-QRA cache, then evaluates plain vs X (each + adaptive conformal):
overall pinball (2024 selection / 2025-26 holdout) and spike-hour metrics.

Usage: python src/qra_x_experiment.py [--workers N]
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import evaluate as ev
from data import DayAheadData
from models import BOACombiner
from probabilistic import MIN_DAYS, QUANTILES, _qra_block, conformalize, pinball, qra

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "qra_x_raw.npy"


def spike_mask(days, y):
    flat = pd.Series(y.ravel(), index=np.repeat(days, 24))
    thr = flat.rolling(24 * 365).quantile(0.975).shift(1).values.reshape(-1, 24)
    return y > thr


def report(name, Q, y, days, spike):
    for lab, m in [("2024 (Auswahl)", (days >= "2024-01-01") & (days <= "2024-12-31")),
                   ("2025-26 (Holdout)", days >= "2025-01-01")]:
        mm = m & np.isfinite(Q).all((1, 2))
        apl = np.mean([pinball(y[mm], Q[mm][..., i], t).mean() for i, t in enumerate(QUANTILES)])
        sp = spike[mm]
        q90 = Q[mm][..., list(QUANTILES).index(0.90)]
        cov_sp = (y[mm] <= q90)[sp].mean()
        apl_sp = np.mean([pinball(y[mm][sp], Q[mm][..., i][sp], t).mean() for i, t in enumerate(QUANTILES)])
        print(f"  {name:28s} {lab:18s} APL {apl:6.3f}  Spike-APL {apl_sp:6.2f}  P(y<=q90|Spike) {cov_sp:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    days, y, P = ev.load()
    F = np.stack([P[m] for m in ev.HYBRID_ALL], -1)

    data = DayAheadData()
    ix = [data.index_of(d) for d in days]
    res = data.tab[ix, :, data.tab_cols.index("res_load")].astype(np.float64) / 1e4
    fc = data.tab[ix, :, data.tab_cols.index("fuel_cost")].astype(np.float64) / 100.0
    Fx = np.concatenate([F, res[..., None], fc[..., None]], -1)

    if OUT.exists():
        Qx = np.load(OUT)
    else:
        print(f"fitting QRA-X ({Fx.shape[-1]} regressors, {len(days)} days, refit daily) ...", flush=True)
        Qx = qra(Fx, y, MIN_DAYS, 1, args.workers)
        np.save(OUT, Qx)
    from probabilistic import cached_qra
    class A: refit_every, workers = 1, args.workers
    Q = cached_qra(F, y, A())                                  # plain QRA (cache hit)

    spike = spike_mask(days, y)
    print("\nRaw QRA:")
    report("QRA alle 13", Q, y, days, spike)
    report("QRA-X (+res_load, fuel)", Qx, y, days, spike)
    print("+ adaptive conformal:")
    Qc = conformalize(Q, y, 2 * MIN_DAYS, gamma=0.005)
    Qxc = conformalize(Qx, y, 2 * MIN_DAYS, gamma=0.005)
    report("QRA + ACI", Qc, y, days, spike)
    report("QRA-X + ACI", Qxc, y, days, spike)

    # DM on daily summed pinball, holdout
    hold = (days >= "2025-01-01") & np.isfinite(Qc).all((1, 2)) & np.isfinite(Qxc).all((1, 2))
    from scipy import stats
    da = np.mean([pinball(y, Qc[..., i], t) for i, t in enumerate(QUANTILES)], 0).sum(1)
    db = np.mean([pinball(y, Qxc[..., i], t) for i, t in enumerate(QUANTILES)], 0).sum(1)
    d = (da - db)[hold]
    print(f"\nDM p(QRA-X+ACI besser, Holdout) = {1 - stats.norm.cdf(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))):.4f}")


if __name__ == "__main__":
    main()
