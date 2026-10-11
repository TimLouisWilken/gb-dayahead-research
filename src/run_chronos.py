"""Zero-shot foundation-model member: Chronos-Bolt (Ansari et al., 2024).

For every delivery day D the pretrained model sees only the hourly price history of the last
28 local days (D-28 .. D-1, all known at the decision time) and forecasts the 24 hours of D.
No training and no exogenous inputs: this tests how far a generic time-series foundation
model gets on GB day-ahead prices. Writes outputs/preds/chronos.npz (point = median, plus quantiles).

Runs in its own environment because chronos needs a transformers version that conflicts with
the main one:
  ~/.cache/day-ahead-prediction/venv-chronos/bin/python src/run_chronos.py
"""
from pathlib import Path

import numpy as np
import torch
from chronos import BaseChronosPipeline

from data import DayAheadData

OUT = Path(__file__).resolve().parents[1] / "outputs" / "preds"
CONTEXT_DAYS, START = 28, "2023-07-01"
TAUS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def live(target):
    """Catch-up + target forecast for live.py: appends to outputs/live_state/preds_chronos.csv.
    Zero-shot, so no state; every past day uses only its own 28-day price history."""
    import pandas as pd
    torch.set_num_threads(4)
    data = DayAheadData(extend_to=target)
    target = pd.Timestamp(target)
    ti = data.index_of(target)
    hist_end = pd.DatetimeIndex(np.load(OUT / "chronos.npz")["days"]).max()
    f = OUT.parent / "live_state" / "preds_chronos.csv"
    done = set(pd.read_csv(f, index_col=0, parse_dates=True).index) if f.exists() else set()
    valid = np.where(data.valid)[0]
    days = np.array([t for t in valid if hist_end < data.days[t] < target and data.days[t] not in done] + [ti])
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)
    ctx = torch.tensor(np.stack([data.y[t - CONTEXT_DAYS:t].ravel() for t in days]), dtype=torch.float32)
    q, _ = pipe.predict_quantiles(ctx, prediction_length=24, quantile_levels=TAUS.tolist())
    pred = q.numpy()[..., list(TAUS).index(0.5)]
    old = pd.read_csv(f, index_col=0, parse_dates=True) if f.exists() else pd.DataFrame(columns=range(24))
    for k, t in enumerate(days):
        old.loc[data.days[t]] = pred[k].astype(float)
    old.sort_index().to_csv(f, index_label="day")
    print(f"[chronos] live: {len(days)} day(s) -> {f.name}")


def main():
    import sys
    if "--live" in sys.argv:
        return live(sys.argv[sys.argv.index("--live") + 1])
    torch.set_num_threads(4)
    data = DayAheadData()
    valid = np.where(data.valid)[0]
    days = valid[valid >= data.index_of(START)]
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.float32)
    ctx = torch.tensor(np.stack([data.y[t - CONTEXT_DAYS:t].ravel() for t in days]), dtype=torch.float32)
    q_all = []
    for b in range(0, len(days), 64):
        q, _ = pipe.predict_quantiles(ctx[b:b + 64], prediction_length=24, quantile_levels=TAUS.tolist())
        q_all.append(q.numpy())
    quant = np.concatenate(q_all)                                  # (n, 24, 19)
    pred = quant[..., list(TAUS).index(0.5)]
    np.savez(OUT / "chronos.npz", idx=days, pred=pred, quant=quant, taus=TAUS,
             days=data.days[days].values.astype("datetime64[D]"))
    print(f"[chronos] done: {len(days)} days, MAE {np.abs(pred - data.y[days]).mean():.2f}")


if __name__ == "__main__":
    main()
