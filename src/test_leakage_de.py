"""Leakage test by perturbation for the German market (MARKET=DE), see test_leakage.py for GB.

For a delivery day D, every value that was *not yet known* at the decision time (10:00
Europe/Berlin on D-1) is replaced by noise; the model inputs for day D must not change:

  prices        local delivery days >= D           (D is the auction being forecast)
  realised load, wind, solar
                UTC hours from local day D-1 00:00  (D-1 is still running). This includes the
                realised generation that calibrates the wind/solar proxies, so the test also
                checks that the weekly proxy refits use only days <= D-2.
  TTF / EUA     settlement days >= D-1             (features use D-2)

Not testable by perturbation, checked by construction:
  load forecast    ENTSO-E day-ahead total load forecast, published by law two hours before
                   gate closure (10:00); SMARD carries no issue times
  weather          previous_day2 = forecast made 48 h before each valid hour, so even the
                   last hour of D was forecast before D-1 00:00 UTC
  TSO wind/solar   not used (DE_RENEW=weather); published after the decision time

Usage: MARKET=DE python src/test_leakage_de.py
"""
import os

os.environ["MARKET"] = "DE"
os.environ["EXTRA_FEATS"] = "2"          # superset for DE (adds wind forecast error of D-2/D-3)
os.environ.setdefault("DE_RENEW", "weather")

import numpy as np
import pandas as pd

import data
import data_de

rng = np.random.default_rng(0)


def local_midnight_utc(day):
    return pd.Timestamp(day).tz_localize(data_de.TZ).tz_convert("UTC").tz_localize(None)


def perturbed_features(day, base, price_shift_days=0):
    orig_smard, orig_fuel = data_de.load_smard, data_de.load_fuel_daily
    cut_price = local_midnight_utc(pd.Timestamp(day) - pd.Timedelta(days=price_shift_days))
    cut_real = local_midnight_utc(pd.Timestamp(day) - pd.Timedelta(days=1))
    d = pd.Timestamp(day)

    def smard():
        s = orig_smard().copy()
        m = s.index >= cut_price
        s.loc[m, "price"] = rng.normal(500, 300, m.sum())
        m = s.index >= cut_real
        for c, (mu, sd) in {"load": (5e4, 2e4), "wind_on": (2e4, 1e4), "wind_off": (4e3, 2e3),
                            "solar": (2e4, 1e4)}.items():
            s.loc[m, c] = rng.normal(mu, sd, m.sum())
        return s

    def fuel():
        f = orig_fuel().copy()
        m = f.index >= d - pd.Timedelta(days=1)
        f.loc[m, "gas"] = rng.normal(80, 30, m.sum())
        f.loc[m, "co2"] = rng.normal(300, 100, m.sum())
        return f

    data_de.load_smard, data_de.load_fuel_daily = smard, fuel
    data_de.renewable_proxies.cache_clear()
    try:
        pert = data.DayAheadData()
    finally:
        data_de.load_smard, data_de.load_fuel_daily = orig_smard, orig_fuel
        data_de.renewable_proxies.cache_clear()
    i, j = base.index_of(day), pert.index_of(day)
    return {k: (getattr(base, k)[i], getattr(pert, k)[j]) for k in ("tab", "hist", "fut")}


if __name__ == "__main__":
    base = data.DayAheadData()
    # incl. both DST switch days, a proxy-refit boundary week and the 15-minute MTU switch
    days = ["2022-03-27", "2023-01-17", "2024-07-15", "2025-10-01", "2025-10-26", "2026-06-21"]
    print("Perturbation test DE (inputs of day D must not change when post-decision data is scrambled):")
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
    print("ALL PASS" if ok_all else "LEAKAGE DETECTED")
