"""Publish the live forecast to the public GitHub-Pages site.

Builds two JSON files from outputs/live/ and pushes them to the site repo
(a plain clone outside iCloud, default ~/.cache/day-ahead-prediction/site):

  data/latest.json    the newest day-ahead forecast: point forecasts + centred QRA-8 quantiles
  data/history.json   daily MAE of the published forecasts vs the naive benchmark
                      (derived statistics only - raw exchange prices are not redistributed)

Called by run_live.sh after live.py; safe to run standalone. Never raises on git/push
problems (the forecast itself must not fail because the website is unreachable).

Usage: python src/publish_site.py [--site DIR] [--no-push]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "outputs" / "live"
SITE = Path.home() / ".cache" / "day-ahead-prediction" / "site"
QCOLS = [f"q{q:02d}" for q in range(5, 100, 5)]


def load_prices_by_day():
    """Actual prices as {UK-local day -> (24,) array}, using the SAME local-day conversion
    as data.py (load_prices is UTC-hourly; naive calendar grouping would shift BST days
    by one hour)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import data as data_mod
    tl = data_mod.to_local_days(data_mod.load_prices())
    return {d: row for d, row in zip(tl.index, tl.values) if np.isfinite(row).all()}


QBAND = ("q05", "q25", "q50", "q75", "q95")
QRA8_LIVE_SINCE = pd.Timestamp("2026-10-27")     # first delivery day published with centred QRA-8


def bands(q):
    """(24, 19) quantile array -> the five band columns shown on the page."""
    return {c: [round(float(v), 2) for v in q[:, i]] for c, i in zip(QBAND, (0, 4, 9, 14, 18))}


def naive(price_by_day, d):
    """Standard EPF naive: Tue-Fri = yesterday, Mon/Sat/Sun = same day last week."""
    ref = d - pd.Timedelta(days=1 if d.dayofweek in (1, 2, 3, 4) else 7)
    return price_by_day.get(ref)


def git(site, *args, timeout=120):
    return subprocess.run(["git", *args], cwd=site, capture_output=True, text=True, timeout=timeout)


def sync(site):
    """Fast-forward the site clone to GitHub before writing. Never raises."""
    try:
        r = git(site, "pull", "--rebase", "--autostash", "-q")
        if r.returncode != 0:
            git(site, "rebase", "--abort")
            print(f"WARNING site pull failed, continuing with the local clone: {r.stderr.strip()[:300]}")
    except Exception as e:
        print(f"WARNING site pull failed: {e}")


def push(site):
    """Push the new data commit. If GitHub moved on in the meantime (a page edit), rebase the
    data commit on top and push again; for files touched on both sides our fresh data wins
    (-X theirs = the commit being replayed)."""
    r = git(site, "push")
    if r.returncode == 0:
        print("pushed to GitHub Pages")
        return
    print(f"push rejected, rebasing onto GitHub and retrying: {r.stderr.strip()[:200]}")
    rb = git(site, "pull", "--rebase", "-X", "theirs", "-q")
    if rb.returncode != 0:
        git(site, "rebase", "--abort")
        print(f"WARNING site rebase failed, forecast NOT published: {rb.stderr.strip()[:300]}")
        return
    r = git(site, "push")
    print("pushed to GitHub Pages (after rebase)" if r.returncode == 0
          else f"WARNING site push failed, forecast NOT published: {r.stderr.strip()[:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=str(SITE))
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()
    site = Path(args.site)
    # bring the site clone up to date first: the page (index.html, classic/) is also edited
    # outside this pipeline, and a push from a stale clone would be rejected as non-fast-forward
    if not args.no_push and (site / ".git").exists():
        sync(site)
    (site / "data").mkdir(parents=True, exist_ok=True)

    files = sorted(LIVE.glob("forecast_*.csv"))
    if not files:
        print("no forecasts yet"); return
    latest_f = files[-1]
    target = latest_f.stem.replace("forecast_", "")
    df = pd.read_csv(latest_f)

    latest = {
        "target_date": target,
        "generated_utc": pd.Timestamp.now("UTC").round("s").isoformat().replace("+00:00", "Z"),
        "timezone": "Europe/London (hour_local)",
        "unit": "EUR/MWh",
        "model": "online hybrid ensemble (BOA, 6 selected members) + centred QRA-8 quantiles + adaptive conformal",
        "hours": df["hour_local"].tolist(),
        "forecast": [round(v, 2) for v in df["boa_select6"]],
        "forecast_all13": [round(v, 2) for v in df["boa_all"]],
        "quantiles": {c: [round(v, 2) for v in df[c]] for c in QCOLS if c in df},
    }
    (site / "data" / "latest.json").write_text(json.dumps(latest))

    # realised accuracy of previously published forecasts (aggregates only)
    try:
        pbd = load_prices_by_day()
    except Exception as e:
        print(f"WARNING prices unavailable ({e}); history not updated")
        pbd = {}
    # --- backtest reconstruction: the research out-of-sample history, combined with the
    # CURRENT system (BOA selected-6, weight smoothing) + centred QRA-8 + ACI quantiles. Curve
    # days before the live era are badged "backtest"; error statistics cover the full eval period.
    bt_days, bt_err = {}, {}
    live_fc = {pd.Timestamp(f.stem.replace("forecast_", "")): pd.read_csv(f) for f in files}
    Qc, qix = None, {}
    try:
        import evaluate as ev
        from models import BOACombiner
        from probabilistic import MIN_DAYS, centre, conformalize
        rdays, ry, P = ev.load()
        pred, _ = ev.run_combiner(BOACombiner(len(ev.HYBRID_SEL)), ev.HYBRID_SEL, P, ry)
        nv_r = P.get("naive")
        z = np.load(ROOT / "outputs" / "live_state" / "qra8_state.npz")
        qd, Qr = pd.DatetimeIndex(z["days"]), z["Q"]
        qix = {d: i for i, d in enumerate(qd)}
        # centre live days on the forecast that was actually published (the first live day was
        # produced on the Mac and differs from the state's re-run point path by up to 14 EUR)
        point = z["point"].copy()
        for d, fc in live_fc.items():
            if d in qix:
                point[qix[d]] = fc["boa_select6"].values
        yq = np.stack([pbd.get(d, np.full(24, np.nan)) for d in qd])
        Qc = conformalize(centre(Qr, point), yq, 2 * MIN_DAYS, gamma=0.005)
        for i, d in enumerate(rdays):
            if d not in pbd:
                continue
            e = float(np.abs(pred[i] - pbd[d]).mean())
            en = float(np.abs(nv_r[i] - pbd[d]).mean()) if nv_r is not None else None
            bt_err[d] = (e, en)
            j = qix.get(d)
            q = Qc[j] if j is not None and np.isfinite(Qc[j]).all() else None
            bt_days[d] = {"date": str(d.date()), "src": "backtest",
                          "actual": [round(float(v), 2) for v in pbd[d]],
                          "forecast": [round(float(v), 2) for v in pred[i]],
                          **(bands(q) if q is not None else {})}
        # gap between the end of the research run and the first live day: the member forecasts
        # exist in the live panel (catch-up runs, predict first, then learn), so these days are
        # reconstructed from the state's point path and badged "backtest" as well
        for d, j in qix.items():
            if d <= rdays[-1] or d in live_fc or d not in pbd or not np.isfinite(Qc[j]).all():
                continue
            nv = naive(pbd, d)
            bt_err[d] = (float(np.abs(point[j] - pbd[d]).mean()),
                         None if nv is None else float(np.abs(nv - pbd[d]).mean()))
            bt_days[d] = {"date": str(d.date()), "src": "backtest",
                          "actual": [round(float(v), 2) for v in pbd[d]],
                          "forecast": [round(float(v), 2) for v in point[j]],
                          **bands(Qc[j])}
    except Exception as e:
        print(f"WARNING backtest reconstruction skipped: {e}")

    hist, days_out = [], []
    live_dates = set()
    for f in files[:-1][-120:] + [latest_f]:
        d = pd.Timestamp(f.stem.replace("forecast_", ""))
        if d not in pbd:
            continue
        y = pbd[d]
        fc = live_fc[d]
        nv = naive(pbd, d)
        # days published with the old QRA-15 band: show the band of the current system instead
        # (centred on the published point forecast), flagged so the page can say so
        j = qix.get(d)
        redo = d < QRA8_LIVE_SINCE and j is not None and Qc is not None and np.isfinite(Qc[j]).all()
        band = bands(Qc[j]) if redo else {c: [round(v, 2) for v in fc[c]] for c in QBAND if c in fc}
        hist.append({"date": str(d.date()),
                     "mae": round(float(np.abs(fc["boa_select6"] - y).mean()), 2),
                     "mae_all13": round(float(np.abs(fc["boa_all"] - y).mean()), 2),
                     "mae_naive": None if nv is None else round(float(np.abs(nv - y).mean()), 2),
                     "cov90": round(float(np.mean((y >= np.array(band["q05"])) & (y <= np.array(band["q95"])))), 3)
                     if {"q05", "q95"} <= set(band) else None})
        # per-day comparison view: only a short recent window of actual auction prices is
        # republished (source: Nord Pool N2EX), the full history stays derived-statistics-only
        live_dates.add(d)
        days_out.append({"date": str(d.date()), "src": "live",
                         "actual": [round(float(v), 2) for v in y],
                         "forecast": [round(v, 2) for v in fc["boa_select6"]],
                         **band, **({"band": "recomputed"} if redo else {})})
    # error statistics: full eval period (backtest) + live days on top
    live_err = {pd.Timestamp(h["date"]): h for h in hist}
    full = []
    for d in sorted(set(bt_err) | set(live_err)):
        if d.year < 2024:
            continue
        if d in live_err:
            full.append(live_err[d])
        else:
            e, en = bt_err[d]
            full.append({"date": str(d.date()), "mae": round(e, 2),
                         "mae_naive": None if en is None else round(en, 2), "src": "backtest"})
    (site / "data" / "history.json").write_text(json.dumps(full))
    # curves: everything from 2024 on. Historical actual prices come from the Ember dataset
    # (CC-BY-4.0, attribution on the page); only the post-Ember Nord Pool top-up days are
    # additionally limited to a rolling 60-day window.
    try:
        import data as data_mod
        e = pd.read_csv(data_mod.RAW / "elec_price_hourly_ember.csv", usecols=["Datetime (UTC)"])
        ember_end = pd.to_datetime(e["Datetime (UTC)"], format="ISO8601").max().normalize() - pd.Timedelta(days=1)
    except Exception:
        ember_end = pd.Timestamp("2000-01-01")
    merged = {pd.Timestamp(r["date"]): r for r in list(bt_days.values()) + days_out}
    cutoff_np = pd.Timestamp.today().normalize() - pd.Timedelta(days=60)
    curves = []
    for d in sorted(merged):
        if d.year < 2024:
            continue
        if d <= ember_end:
            merged[d]["price_src"] = "ember"
        elif d >= cutoff_np:
            merged[d]["price_src"] = "nordpool"
        else:
            continue
        curves.append(merged[d])
    (site / "data" / "days.json").write_text(json.dumps(curves))
    print(f"history: {len(full)} day(s), curves: {len(curves)} day(s)")
    print(f"site data written: {target}, {len(hist)} evaluated day(s) -> {site/'data'}")

    if args.no_push or not (site / ".git").exists():
        return
    try:
        subprocess.run(["git", "add", "-A"], cwd=site, check=True, capture_output=True)
        r = subprocess.run(["git", "commit", "-m", f"forecast {target}"], cwd=site, capture_output=True, text=True)
        if r.returncode == 0:
            push(site)
        else:
            print("nothing new to commit")
    except Exception as e:
        print(f"WARNING site push failed: {e}")


if __name__ == "__main__":
    main()
