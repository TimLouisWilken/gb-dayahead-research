"""Rolling-origin online simulation of one forecasting member.

For every delivery day t from START to the end of the data the member forecasts
day t using only days < t for training (prices of day t-1 are known on the
morning of t-1, because they cleared in the D-2 auction). Members are retrained
or updated on their own cadence:

  naive        no training
  lear         full recalibration every day (182- and 728-day windows, averaged)
  rf_long      refit every 7 days on 3 years, exponential recency weights (half-life 365 d)
  rf_short     refit every day on the last 84 days
  rf_static    fitted once at START, never updated (ablation)
  lstm         cold fit at START, warm-start fine-tuning every day with experience
               replay, full cold retrain every 91 days
  lstm_static  fitted once at START, never updated (ablation)
  lstm_rf      LSTM + RF error correction; reads the lstm forecasts, refits weekly
  tcn, transformer, dnn
               other network architectures, same continual-learning scheme as lstm
  ddnn         distributional LSTM (Johnson SU output), same scheme; also stores quantiles
  gbm          histogram gradient boosting (absolute loss), refit every 7 days on 3 years
  lear4        LEAR averaged over 4 calibration windows (56, 84, 1092, 1456 days), daily
  merit        merit-order regression on residual load x fuel cost, daily on 182 days
  chronos      zero-shot foundation model, see run_chronos.py (separate environment)

Each member runs in its own process (keeps torch and sklearn OpenMP runtimes
apart) and writes outputs/preds/<member>.npz.

Usage: python src/run_online.py <member> [--start 2023-07-01]
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from data import DayAheadData

OUT = Path(__file__).resolve().parents[1] / "outputs" / "preds"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("member")
    ap.add_argument("--start", default="2023-07-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output name (default: the member name); "
                    "used to rerun a member after a feature change without overwriting")
    ap.add_argument("--base", default="lstm", help="lstm_rf only: name of the base forecasts")
    args = ap.parse_args()

    data = DayAheadData()
    valid = np.where(data.valid)[0]
    s = data.index_of(args.start)
    e = data.index_of(args.end) + 1 if args.end else len(data.days)
    days = valid[(valid >= s) & (valid < e)]
    preds = np.full((len(days), 24), np.nan, dtype=np.float64)
    m = args.member
    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    neural = {"lstm": {}, "lstm_static": {}, "tcn": {"arch": "tcn"}, "transformer": {"arch": "transformer"},
              "dnn": {"arch": "mlp"}, "ddnn": {"dist": "jsu"}}
    quants = None
    if m in neural:
        import torch
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
        torch.manual_seed(args.seed)
        from models import NeuralMember
        make = lambda: NeuralMember(n_seeds=3, seed=args.seed, **{**LSTM_CFG, **neural[m]})
        if m == "ddnn":
            quants = np.full((len(days), 24, len(TAUS)), np.nan)
    else:
        from models import GBMember, LEARMember, MeritOrderMember, RFMember, ResidualRF, naive_forecast

    if m == "lstm_rf":
        base = dict(np.load(OUT / f"{args.base}.npz"))
        base_pred = dict(zip(base["idx"].tolist(), base["pred"]))

    model, last_fit = None, -10**9
    for k, t in enumerate(days):
        train = valid[valid < t]
        if m == "naive":
            preds[k] = naive_forecast(data, t)[0]
            continue
        if m == "lear":
            model = LEARMember().fit(data, train)
        elif m == "rf_long":
            if k - last_fit >= 7:
                model, last_fit = RFMember(window=1092, half_life=365, seed=args.seed).fit(data, train), k
        elif m == "rf_short":
            model = RFMember(window=84, seed=args.seed).fit(data, train)
        elif m == "rf_static":
            if model is None:
                model = RFMember(window=1092, half_life=365, seed=args.seed).fit(data, train)
        elif m == "lstm_static":
            if model is None:
                model = make().fit(data, train)
        elif m == "gbm":
            if k - last_fit >= 7:
                model, last_fit = GBMember(seed=args.seed).fit(data, train), k
        elif m in ("gbm_q10", "gbm_q90"):            # tail members (spike information for QRA)
            if k - last_fit >= 7:
                q = 0.1 if m.endswith("q10") else 0.9
                model, last_fit = GBMember(quantile=q, seed=args.seed).fit(data, train), k
        elif m == "lear4":
            model = LEARMember(windows=(56, 84, 1092, 1456)).fit(data, train)
        elif m == "merit":
            model = MeritOrderMember().fit(data, train)
        elif m in ("lstm", "tcn", "transformer", "dnn", "ddnn"):
            if k - last_fit >= 91:
                model, last_fit = make().fit(data, train), k
            else:
                model.update(data, train, rng)
        elif m == "lstm_rf":
            hist = np.array([i for i in train if i in base_pred])
            if len(hist) < 28:                      # not enough out-of-sample errors yet
                preds[k] = base_pred[t]
                continue
            if k - last_fit >= 7:
                model, last_fit = ResidualRF(seed=args.seed).fit(
                    data, hist, np.stack([base_pred[i] for i in hist])), k
            preds[k] = model.predict(data, t, base_pred[t][None])[0]
            continue
        else:
            raise SystemExit(f"unknown member {m}")
        preds[k] = model.predict(data, t)[0]
        if quants is not None:
            quants[k] = model.predict_quantiles(data, t, TAUS)[0]

        if k % 30 == 0:
            done = slice(0, k + 1)
            mae = np.nanmean(np.abs(preds[done] - data.y[days[done]]))
            print(f"[{m}] {data.days[t].date()}  {k + 1}/{len(days)}  running MAE {mae:6.2f}  "
                  f"{time.time() - t_start:6.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    extra = {"quant": quants, "taus": TAUS} if quants is not None else {}
    name = args.out or m
    np.savez(OUT / f"{name}.npz", idx=days, pred=preds, days=data.days[days].values.astype("datetime64[D]"), **extra)
    mae = np.nanmean(np.abs(preds - data.y[days]))
    print(f"[{m}] done: {len(days)} days, MAE {mae:.2f}, {time.time() - t_start:.0f}s")


TAUS = np.round(np.arange(0.05, 0.96, 0.05), 2)     # quantile levels of the distributional member

# Neural-net hyper-parameters chosen on the 2023-H2 burn-in period (see README); shared by all archs
LSTM_CFG = dict(hidden=64, revin=True, half_life=365)

if __name__ == "__main__":
    main()
