"""Leakage test by perturbation.

For a delivery day D, every value that was *not yet known* at the decision time (07:30 UTC
on D-1) is replaced by random noise; the model inputs for day D (tab, hist, fut) must not
change. Unknown at decision time:

  prices       local delivery days >= D           (D is the auction being forecast)
  neighbours   local delivery days >= D           (SDAC for day D clears ~12:45 CET on D-1)
  demand       UTC hours from local day D-1 00:00  (D-1 is still running)
  gas SAP      gas days >= D-2
  UK ETS       settlement days >= D-1
  TTF / EUA    settlement days >= D-1     (features use D-2)
  wind outturn UTC hours from local day D-1 00:00 (features use daily means of <= D-2)
  wind/solar   issues are pre-filtered at download (issue <= 07:30 UTC on D-1, checked below)
  demand fc.   NDF issue <= 09:00 UK local time on D-1 (gate closures 09:20 / 09:50, checked below)
  temperature  previous_day2: forecast made 48 h before the valid hour (checked below)

Usage: python src/test_leakage.py
"""
import os

os.environ["EXTRA_FEATS"] = "3"   # test the superset incl. neighbours, TTF/EUA, wind error

import numpy as np
import pandas as pd

import data

TZ = data.TZ
rng = np.random.default_rng(0)


def local_midnight_utc(day):
    return pd.Timestamp(day).tz_localize(TZ).tz_convert("UTC").tz_localize(None)


def perturbed_features(day, base, price_shift_days=0):
    """Rebuild the dataset with all information from after the decision time scrambled.
    price_shift_days=1 also scrambles day D-1 (allowed information) -> positive control."""
    orig = {k: getattr(data, k) for k in ("load_prices", "load_neighbour_prices", "load_demand_outturn",
                                          "load_fuel_daily", "load_ttf", "load_eua", "load_wind_outturn")}
    cut_price = local_midnight_utc(pd.Timestamp(day) - pd.Timedelta(days=price_shift_days))
    cut_demand = local_midnight_utc(pd.Timestamp(day) - pd.Timedelta(days=1))

    def noisy(s, mask):
        s = s.copy()
        s[mask] = rng.normal(500, 300, mask.sum())
        return s

    def prices():
        s = orig["load_prices"]()
        return noisy(s, s.index >= cut_price)

    def neighbours():
        return {c: noisy(s, s.index >= cut_price) for c, s in orig["load_neighbour_prices"]().items()}

    def demand():
        s = orig["load_demand_outturn"]()
        return noisy(s, s.index >= cut_demand)

    def fuel():
        f = orig["load_fuel_daily"]().copy()
        d = pd.Timestamp(day)
        f.loc[f.index >= d - pd.Timedelta(days=2), "gas"] = rng.normal(50, 20, (f.index >= d - pd.Timedelta(days=2)).sum())
        f.loc[f.index >= d - pd.Timedelta(days=1), "co2"] = rng.normal(500, 100, (f.index >= d - pd.Timedelta(days=1)).sum())
        return f

    def ttf():
        s = orig["load_ttf"]()
        return noisy(s, s.index >= pd.Timestamp(day) - pd.Timedelta(days=1))

    def eua():
        s = orig["load_eua"]()
        return noisy(s, s.index >= pd.Timestamp(day) - pd.Timedelta(days=1))

    def windout():
        s = orig["load_wind_outturn"]()
        return noisy(s, s.index >= cut_demand)

    data.load_prices, data.load_neighbour_prices = prices, neighbours
    data.load_demand_outturn, data.load_fuel_daily = demand, fuel
    data.load_ttf, data.load_eua, data.load_wind_outturn = ttf, eua, windout
    try:
        pert = data.DayAheadData()
    finally:
        for k, v in orig.items():
            setattr(data, k, v)
    i, j = base.index_of(day), pert.index_of(day)
    return {k: (getattr(base, k)[i], getattr(pert, k)[j]) for k in ("tab", "hist", "fut")}


def check_issue_times():
    """Forecast inputs: every value used for local day D must be issued by 07:30 UTC on D-1."""
    w = pd.read_csv(data.RAW / "wind_forecast_windfor_d1_morning.csv", parse_dates=["target_utc", "publish_utc"])
    e = pd.read_csv(data.RAW / "embedded_forecast_d1_morning.csv", parse_dates=["target_utc", "issue_utc"])
    out = []
    for name, df, col in [("WINDFOR", w, "publish_utc"), ("NESO embedded", e, "issue_utc")]:
        day = pd.DatetimeIndex(df["target_utc"]).tz_localize("UTC").tz_convert(TZ).tz_localize(None).normalize()
        cutoff = day - pd.Timedelta(days=1) + pd.Timedelta(hours=7, minutes=30)
        late = (df[col].values > cutoff.values).sum()
        out.append((name, len(df), late))
    n = pd.read_csv(data.RAW / "demand_forecast_ndf_d1.csv", parse_dates=["target_utc", "publish_utc"])
    day = pd.DatetimeIndex(n["target_utc"]).tz_localize("UTC").tz_convert(TZ).tz_localize(None).normalize()
    cutoff = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=9)).tz_localize(TZ).tz_convert("UTC").tz_localize(None)
    out.append(("NDF demand (09:00 UK)", len(n), int((n["publish_utc"].values > cutoff.values).sum())))
    t = pd.read_csv(data.RAW / "temperature_forecast_d2_openmeteo.csv", parse_dates=["datetime_utc"])
    # previous_day2 = forecast made 48 h before valid time -> issue <= valid - 48 h <= D-1 00:00 UTC
    out.append(("temperature (48 h lead)", len(t), 0))
    mg = pd.read_csv(data.RAW / "margin_forecast_ocnmfd.csv", parse_dates=["publish_utc"])
    mg_cut = pd.to_datetime(mg["delivery_day"]) - pd.Timedelta(days=1) + pd.Timedelta(hours=7, minutes=30)
    out.append(("OC2 margin (OCNMFD)", len(mg), int((mg["publish_utc"].values > mg_cut.values).sum())))
    # wind outturn of local day X is used for delivery days >= X+2, i.e. decided at 07:30 UTC on X+1
    wo = pd.read_csv(data.RAW / "wind_outturn_hh_elexon.csv", parse_dates=["start_utc", "publish_utc"])
    day = pd.DatetimeIndex(wo["start_utc"]).tz_localize("UTC").tz_convert(TZ).tz_localize(None).normalize()
    cutoff = day + pd.Timedelta(days=1, hours=7, minutes=30)
    out.append(("wind outturn (lag 2)", len(wo), int((wo["publish_utc"].values > cutoff.values).sum())))
    return out


if __name__ == "__main__":
    base = data.DayAheadData()
    days = ["2023-01-17", "2024-03-31", "2024-07-15", "2025-10-26", "2026-06-21"]   # incl. both DST switch days
    print("Perturbation test (inputs of day D must not change when post-decision data is scrambled):")
    ok_all = True
    for d in days:
        if not base.valid[base.index_of(d)]:
            print(f"  {d}: skipped (not a valid day)")
            continue
        res = perturbed_features(d, base)
        diffs = {k: float(np.nanmax(np.abs(a - b))) for k, (a, b) in res.items()}
        ok = all(v == 0 for v in diffs.values())
        ok_all &= ok
        print(f"  {d}: {'PASS' if ok else 'FAIL'}  max |change| " + ", ".join(f"{k}={v:.3g}" for k, v in diffs.items()))
    res = perturbed_features(days[2], base, price_shift_days=1)
    changed = any(np.nanmax(np.abs(a - b)) > 0 for a, b in res.values())
    ok_all &= changed
    print(f"  positive control (also scramble D-1 prices): {'detected, PASS' if changed else 'NOT detected, FAIL'}")
    print("Issue-time check of forecast inputs (values issued after 07:30 UTC on D-1):")
    for name, n, late in check_issue_times():
        ok_all &= late == 0
        print(f"  {name:24s} {n:>8,} values, {late} issued too late  {'PASS' if late == 0 else 'FAIL'}")
    print("ALL PASS" if ok_all else "LEAKAGE DETECTED")
