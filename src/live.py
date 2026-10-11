"""Daily live forecast: tomorrow's 24 hourly GB day-ahead prices, using the online members
with persistent state.

What one run does (intended schedule: ~08:10 UK time on D-1, after the 07:30 UTC data cutoff
and before the EPEX/N2EX gate closures at 09:20/09:50 UK time):

  1. refresh all inputs, including three live top-ups that bridge the lag of the bulk sources
     (Nord Pool N2EX auction results, Elexon INDO demand, NESO embedded datastore)
  2. per member: catch up the rolling-origin simulation over any days since the last run
     (predict first, then learn - exactly the protocol of run_online.py) and forecast the
     target day; neural members keep their fine-tuned weights in outputs/live_state/*.pkl,
     cold-refitting every 91 days as in the research runs
  3. combine: BOA over all members (weights warm-started from the full out-of-sample history
     since 2023-07), the ablation-selected 6-member BOA, and equal weights
  4. quantiles: QRA over all members + adaptive conformal (the exported product of
     probabilistic.py), extended incrementally from a persistent QRA state

Output: outputs/live/forecast_<target>.csv (+ a row appended to outputs/live/history.csv).
Every member forecast for a past day is genuinely out-of-sample (made before its price was
used for anything), so the live history doubles as a continuing evaluation set.

Usage:
  python src/live.py                     # target = tomorrow, full update + forecast
  python src/live.py --target 2026-10-11 --skip-update
  python src/live.py --worker gbm --target 2026-10-11   # internal: one member's catch-up
"""
import argparse
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "outputs" / "preds"
STATE = ROOT / "outputs" / "live_state"
OUT = ROOT / "outputs" / "live"
PY = sys.executable
CHRONOS_PY = Path(os.environ.get("CHRONOS_PY",
                  str(Path.home() / ".cache" / "day-ahead-prediction" / "venv-chronos" / "bin" / "python")))

# the 13 members of "Hybrid BOA all models" (order = evaluate.HYBRID_ALL) + naive reference
MEMBERS = ["rf_long", "rf_short", "lstm", "lstm_rf", "gbm", "tcn", "transformer", "dnn", "ddnn",
           "lear", "lear4", "merit", "chronos"]
# ablation-selected subset (backward elimination on 2024 only, validated on 2025-26: MAE 9.18
# vs 9.47 of the full set, DM p = 0.002): one member per model family
SELECT6 = ["lstm_rf", "gbm", "transformer", "lear", "merit", "chronos"]
# dedicated GBM tail forecasts (pinball loss, EXTRA_FEATS=2): QRA regressors only, never in BOA.
# Selected on 2024 / confirmed on the 2025-26 holdout (DM p = 0.038, spike q90 coverage 65->68 %).
TAILS = ["gbm_q10", "gbm_q90"]
QRA_MEMBERS = MEMBERS + TAILS
NEURAL = {"lstm": {}, "tcn": {"arch": "tcn"}, "transformer": {"arch": "transformer"},
          "dnn": {"arch": "mlp"}, "ddnn": {"dist": "jsu"}}
LSTM_CFG = dict(hidden=64, revin=True, half_life=365)      # as in run_online.py
UPDATE_STEPS = ["prices_live", "demand_live", "embedded_live", "windfor", "ndf",
                "ttf", "windout", "weather", "gas", "carbon"]
QUANTILES = np.round(np.arange(0.05, 0.96, 0.05), 2)
QRA_WINDOW, MIN_DAYS, ACI_GAMMA = 182, 56, 0.005           # as in probabilistic.py


def hist_preds(m):
    """Date-indexed out-of-sample forecasts of the frozen research run (outputs/preds)."""
    z = np.load(PRED / f"{m}.npz")
    return pd.DataFrame(z["pred"], index=pd.DatetimeIndex(z["days"]))


def live_preds(m):
    f = STATE / f"preds_{m}.csv"
    if not f.exists():
        return pd.DataFrame()
    return pd.read_csv(f, index_col=0, parse_dates=True)


def save_live_pred(m, day, pred):
    df = live_preds(m)
    pred = np.asarray(pred, dtype=float).ravel()
    if df.empty:
        df = pd.DataFrame(columns=range(len(pred)))
    df.loc[pd.Timestamp(day)] = pred
    df.sort_index().to_csv(STATE / f"preds_{m}.csv", index_label="day")


def member_panel(m):
    """Full forecast history of a member: research run + live days (live wins on overlap)."""
    h, l = hist_preds(m), live_preds(m)
    if len(l):
        l.columns = h.columns if len(h) else l.columns
        h = pd.concat([h[~h.index.isin(l.index)], l]).sort_index()
    return h


# ------------------------------------------------------------------ one member's catch-up
def worker(m, target):
    from data import DayAheadData
    data = DayAheadData(extend_to=target)
    target = pd.Timestamp(target)
    ti = data.index_of(target)
    if not data.feat_ok[ti]:
        bad = [c for j, c in enumerate(data.tab_cols) if not np.isfinite(data.tab[ti, :, j]).all()]
        raise SystemExit(f"[{m}] target {target.date()} not forecastable, NaN in: {bad}")
    valid = np.where(data.valid)[0]
    hist_end = hist_preds(m).index.max()
    st = {}
    sf = STATE / f"{m}.pkl"
    if sf.exists():
        with open(sf, "rb") as fh:
            st = pickle.load(fh)
    done = set(live_preds(m).index)
    todo = [t for t in valid if hist_end < data.days[t] < target and data.days[t] not in done]
    todo.append(ti)                                         # the target day itself (y unknown)
    t0 = time.time()

    if m in NEURAL:
        import torch
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "3")))
        from models import NeuralMember
        make = lambda: NeuralMember(n_seeds=3, **{**LSTM_CFG, **NEURAL[m]})
    else:
        from models import GBMember, LEARMember, MeritOrderMember, RFMember, ResidualRF, naive_forecast
    if m == "lstm_rf":
        base = pd.concat([hist_preds("lstm"), live_preds("lstm").set_axis(range(24), axis=1)])
        base = base[~base.index.duplicated(keep="last")].sort_index()
        base_pred = {d: v for d, v in zip(base.index, base.values)}

    for t in todo:
        day = data.days[t]
        train = valid[valid < t]
        rng = np.random.default_rng(int(day.strftime("%Y%m%d")) + len(m))
        model, last_fit = st.get("model"), st.get("last_fit")
        age = (day - last_fit).days if last_fit is not None else 10**6
        if m == "naive":
            pred = naive_forecast(data, t)[0]
        elif m in ("lear", "lear4", "merit", "rf_short"):    # daily refits, nothing persists
            model = {"lear": lambda: LEARMember(),
                     "lear4": lambda: LEARMember(windows=(56, 84, 1092, 1456)),
                     "merit": lambda: MeritOrderMember(),
                     "rf_short": lambda: RFMember(window=84)}[m]()
            pred = model.fit(data, train).predict(data, t)[0]
        elif m in ("rf_long", "gbm", "gbm_q10", "gbm_q90"):
            if model is None or age >= 7:
                cls = RFMember(window=1092, half_life=365) if m == "rf_long" \
                    else GBMember(quantile={"gbm_q10": 0.1, "gbm_q90": 0.9}.get(m))
                model = cls.fit(data, train)
                st.update(model=model, last_fit=day)
            pred = model.predict(data, t)[0]
        elif m in NEURAL:
            torch.manual_seed(int(day.strftime("%Y%m%d")))
            if model is None or age >= 91:                   # cold retrain, as every 91 days online
                print(f"[{m}] cold fit on {len(train)} days ...", flush=True)
                model = make().fit(data, train)
                st.update(model=model, last_fit=day, updated_through=day)
            elif day > st.get("updated_through", day - pd.Timedelta(days=1)):
                model.update(data, train, rng)               # once per new day, never on a re-run
                st["updated_through"] = day
            pred = model.predict(data, t)[0]
            if m == "ddnn":
                save_live_pred("ddnn_quant", day, model.predict_quantiles(data, t, QUANTILES)[0].ravel())
        elif m == "lstm_rf":
            hist = np.array([i for i in train if data.days[i] in base_pred])
            if data.days[t] not in base_pred:
                raise SystemExit(f"[{m}] no base lstm forecast for {day.date()}")
            if model is None or age >= 7:
                model = ResidualRF().fit(data, hist, np.stack([base_pred[data.days[i]] for i in hist]))
                st.update(model=model, last_fit=day)
            pred = model.predict(data, t, base_pred[data.days[t]][None])[0]
        else:
            raise SystemExit(f"unknown member {m}")
        save_live_pred(m, day, pred)
        known = "" if t == ti else f"  (actual mean {np.nanmean(data.y[t]):.1f})"
        print(f"[{m}] {day.date()}  mean {np.mean(pred):7.2f}{known}", flush=True)

    if st.get("model") is not None:
        with open(sf, "wb") as fh:
            pickle.dump(st, fh)
    print(f"[{m}] done, {len(todo)} day(s), {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ combination + quantiles
def boa_over(panel, y_map, members, dates, target):
    """Run BOA over all dates with known prices, return the target combination + final weights."""
    from models import BOACombiner
    comb = BOACombiner(len(members))
    F = np.stack([panel[m].loc[dates].values for m in members], -1)       # (n, 24, K)
    for i, d in enumerate(dates):
        if d == target:
            return comb.combine(F[i]), comb.w
        comb.update(F[i], y_map[d])
    raise ValueError("target not in panel")


def qra_quantiles(panel, y_map, dates, target, workers):
    """QRA over all members + adaptive conformal for the target day, with persistent raw-QRA
    state (incremental daily extension; the full-history bootstrap runs once)."""
    sys.path.insert(0, str(Path(__file__).parent))
    from probabilistic import _qra_block, conformalize
    F = np.stack([panel[m].loc[dates].values for m in QRA_MEMBERS], -1)
    y = np.stack([y_map.get(d, np.full(24, np.nan)) for d in dates])
    sf = STATE / "qra15_state.npz"
    Q = np.full((len(dates), 24, len(QUANTILES)), np.nan)
    have = 0
    if sf.exists():
        z = np.load(sf)
        old = pd.DatetimeIndex(z["days"])
        keep = old.isin(dates)
        pos = [dates.get_loc(d) for d in old[keep]]
        Q[pos] = z["Q"][keep]
        # never reuse rows of still-open days (target, days with incomplete prices): their
        # member forecasts change between runs as fresher inputs arrive, so refit them
        Q[~np.isfinite(y).all(1)] = np.nan
        have = int(np.isfinite(Q).all((1, 2)).sum())
    missing = [i for i in range(MIN_DAYS, len(dates)) if not np.isfinite(Q[i]).any()
               and np.isfinite(F[max(0, i - QRA_WINDOW):i]).all()]
    if missing:
        print(f"[qra] fitting {len(missing)} day(s) (bootstrap={have == 0}) ...", flush=True)
        from concurrent.futures import ProcessPoolExecutor
        blocks = np.array_split(np.array(missing), min(workers, len(missing)))
        with ProcessPoolExecutor(workers) as ex:
            for part in ex.map(_qra_block, [(F, y, b, 1) for b in blocks if len(b)]):
                for t, q in part.items():
                    Q[t] = np.sort(q, axis=-1)
    np.savez(sf, days=dates.values.astype("datetime64[D]"), Q=Q)
    Qc = conformalize(Q, y, 2 * MIN_DAYS, gamma=ACI_GAMMA)
    return Qc[dates.get_loc(target)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None, help="delivery day (default: tomorrow)")
    ap.add_argument("--skip-update", action="store_true")
    ap.add_argument("--worker", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    target = pd.Timestamp(args.target) if args.target else pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    STATE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.worker:
        return worker(args.worker, target)

    if not args.skip_update:
        import update_data
        for step in UPDATE_STEPS:
            try:
                update_data.STEPS[step]()
            except Exception as e:                       # a stale slow-moving input is tolerable
                print(f"WARNING {step}: {e}")

    from data import DayAheadData
    data = DayAheadData(extend_to=target)
    ti = data.index_of(target)
    if not data.feat_ok[ti]:
        bad = [c for j, c in enumerate(data.tab_cols) if not np.isfinite(data.tab[ti, :, j]).all()]
        raise SystemExit(f"target {target.date()} not forecastable yet - NaN in {bad}; "
                         "inputs publish by ~07:30 UTC on D-1")
    print(f"\n=== live forecast for {target.date()} ===")

    # members in parallel subprocesses (torch and sklearn OpenMP runtimes kept apart);
    # lstm_rf needs the lstm forecasts, so it runs in a second wave
    def spawn(m):
        env = {**os.environ, "OMP_NUM_THREADS": "3", "TORCH_THREADS": "3"}
        if m.startswith("gbm"):
            env["EXTRA_FEATS"] = "2"                     # adopted after the online A/B test
        py, script = (str(CHRONOS_PY), ["run_chronos.py", "--live", str(target.date())]) if m == "chronos" \
            else (PY, ["live.py", "--worker", m, "--target", str(target.date())])
        return subprocess.Popen([py, "-W", "ignore"] + script, env=env, cwd=Path(__file__).parent)
    wave1 = [m for m in QRA_MEMBERS if m != "lstm_rf"]
    if not CHRONOS_PY.exists():
        wave1.remove("chronos")
        print("WARNING: chronos venv missing -> member skipped")
    for grp in (wave1, ["lstm_rf"]):
        procs = {m: spawn(m) for m in grp}
        fail = [m for m, p in procs.items() if p.wait() != 0]
        if fail:
            raise SystemExit(f"member(s) failed: {fail}")

    # combination over the common forecast history
    panel = {m: member_panel(m) for m in QRA_MEMBERS if (STATE / f"preds_{m}.csv").exists() or (PRED / f"{m}.npz").exists()}
    dates = sorted(set.intersection(*[set(p.index) for p in panel.values()]))
    dates = pd.DatetimeIndex([d for d in dates if d <= target])
    y_map = {d: data.y[data.index_of(d)].astype(np.float64) for d in dates
             if d <= data.days[-1] and np.isfinite(data.y[data.index_of(d)]).all()}
    past = [d for d in dates if d in y_map and d < target]
    print(f"forecast history: {dates[0].date()} -> {dates[-1].date()} ({len(past)} evaluated days)")

    boa_all, w_all = boa_over(panel, y_map, MEMBERS, dates, target)
    boa_sel, _ = boa_over(panel, y_map, SELECT6, dates, target)
    eq_all = np.mean([panel[m].loc[target].values for m in MEMBERS], 0)
    quant = qra_quantiles(panel, y_map, dates, target, args.workers)

    # output file: one row per hour of the local delivery day
    out = pd.DataFrame({"hour_local": range(24),
                        "boa_all": boa_all, "boa_select6": boa_sel, "eq_all": eq_all})
    for m in MEMBERS:
        out[m] = panel[m].loc[target].values
    for i, q in enumerate(QUANTILES):
        out[f"q{int(q * 100):02d}"] = quant[:, i]
    f = OUT / f"forecast_{target.date()}.csv"
    out.round(2).to_csv(f, index=False)

    hist_f = OUT / "history.csv"
    row = pd.DataFrame([{"target": target.date(), "run_utc": pd.Timestamp.now("UTC").round("s").tz_convert(None),
                         **{f"boa_all_h{h}": round(float(boa_all[h]), 2) for h in range(24)}}])
    row.to_csv(hist_f, mode="a", header=not hist_f.exists(), index=False)

    wm = pd.Series(w_all.mean(0), index=MEMBERS).sort_values(ascending=False)
    print(f"\nforecast {target.date()} (EUR/MWh): mean {boa_all.mean():.1f}  "
          f"min {boa_all.min():.1f} (h{boa_all.argmin()})  max {boa_all.max():.1f} (h{boa_all.argmax()})")
    print(out.round(1).to_string(index=False))
    print("\nBOA weights (mean over hours):", ", ".join(f"{k} {v:.2f}" for k, v in wm.items() if v >= 0.01))
    print(f"-> {f}")
    if target in y_map:    # price already published (run after ~10:45 UK on D-1): instant verification
        y = y_map[target]
        print(f"\nauction already published - realised MAE: BOA all {np.abs(boa_all - y).mean():.2f}  "
              f"select6 {np.abs(boa_sel - y).mean():.2f}  eq {np.abs(eq_all - y).mean():.2f}")


if __name__ == "__main__":
    main()
