"""Raw loaders for the German (DE-LU) day-ahead market, used by data.py when MARKET=DE.

Forecasting set-up: before the SDAC day-ahead auction closes (12:00 CET/CEST on D-1) the 24
hourly prices of German local delivery day D are forecast. Decision time: 10:00 Europe/Berlin
on D-1. Information set:

  * DE-LU day-ahead prices of local days <= D-1 (D-1 cleared at ~12:45 on D-2)
  * day-ahead total load forecast for D (ENTSO-E/SMARD; legally published two hours before
    gate closure, i.e. by 10:00)
  * wind and solar for D: the TSO day-ahead forecasts are NOT used by default, because they
    are published later in the day (legal deadline 18:00 D-1; on 2026-10-20 they were still
    missing on SMARD at 10:54 CEST). Instead, wind_da / solar_da are proxies built from weather
    forecasts made 48 h before each valid hour (JMA via Open-Meteo previous runs), calibrated
    causally on realised generation of days <= D-2 (see renewable_proxies). DE_RENEW=tso
    switches to the TSO forecasts (the information set of the open EPF benchmark), for
    comparison only.
  * temperature forecast made 48 h ahead, population-weighted over six regional centres
  * realised load and wind of local days <= D-2
  * TTF front-month and EUA settlements of trading day D-2

All timestamps are UTC hour starts; data.py maps them to local days.
"""
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "de"
RENEW = os.environ.get("DE_RENEW", "weather")           # "weather" (strict, default) or "tso"
TZ = "Europe/Berlin"


@lru_cache(maxsize=1)
def load_smard():
    return pd.read_csv(RAW / "smard_hourly.csv", parse_dates=["datetime_utc"]).set_index("datetime_utc")


@lru_cache(maxsize=1)
def load_weather():
    return pd.read_csv(RAW / "weather_d2_de.csv", parse_dates=["datetime_utc"]).set_index("datetime_utc")


def load_prices():
    return load_smard()["price"].dropna().rename("price")


def load_demand_forecast():
    return load_smard()["load_fc"].dropna().rename("demand_fc")


def load_demand_outturn():
    return load_smard()["load"].dropna().rename("demand")


def load_temperature():
    return load_weather()["temp_pop"].dropna().rename("temperature")


def load_wind_outturn():
    s = load_smard()
    return (s["wind_on"] + s["wind_off"]).dropna().rename("wind_out")


def load_fuel_daily():
    """Daily TTF (EUR/MWh_th) and EUA (EUR/t); weekends/holidays carry the last settlement."""
    ttf = pd.read_csv(RAW / "gas_ttf_daily_yahoo.csv", parse_dates=["Date"]).set_index("Date")["ttf"]
    eua = pd.read_csv(RAW / "carbon_eua_daily_yahoo.csv", parse_dates=["Date"]).set_index("Date")["eua"]
    days = pd.date_range(min(ttf.index.min(), eua.index.min()), max(ttf.index.max(), eua.index.max()), freq="D")
    return pd.concat([ttf.rename("gas"), eua.rename("co2")], axis=1).reindex(days).ffill()


# ---------------------------------------------------------------- wind / solar proxies
def _clear_sky_ghi(t_utc, lat, lon):
    """Haurwitz clear-sky GHI (W/m2) at the middle of each UTC hour."""
    t = pd.DatetimeIndex(t_utc) + pd.Timedelta(minutes=30)
    doy = t.dayofyear.values
    hour = t.hour.values + t.minute.values / 60
    g = 2 * np.pi * (doy - 1) / 365
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))     # minutes
    ha = np.radians(15 * (hour + (eot + 4 * lon) / 60 - 12))
    la = np.radians(lat)
    cosz = np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)
    return np.where(cosz > 0.01, 1098 * cosz * np.exp(-0.057 / np.maximum(cosz, 0.01)), 0.0)


def _wind_design(w):
    """Per wind point: hub-height speed and a generic power curve (cut-in 3, rated 12, cut-out 25 m/s)."""
    cols = {}
    for c in [c for c in w.columns if c.startswith("wind_")]:
        v = w[c].values / 3.6 * (1.25 if "off" in c else 1.45)           # 10 m -> hub height
        cf = np.clip((v ** 3 - 27) / (12 ** 3 - 27), 0, 1) * (v < 25)
        cols[f"cf_{c}"], cols[f"v_{c}"] = cf, v
    return pd.DataFrame(cols, index=w.index)


def _solar_design(w):
    from update_data_de import PV_POINTS
    cols, clear = {}, []
    for c, (lat, lon) in PV_POINTS.items():
        ghi = _clear_sky_ghi(w.index, lat, lon)
        cc = w[f"cc_{c}"].values / 100
        cols[f"pv_{c}"] = ghi * (1 - 0.75 * cc ** 3.4)                    # Kasten-Czeplak
        clear.append(ghi)
    cols["clear"] = np.mean(clear, 0)
    return pd.DataFrame(cols, index=w.index)


def _rolling_fit(X, y, local_day, window, refit=7, half_life=None, positive_cols=None):
    """Weekly ridge refits; block starting on local day W is fitted on hours of days <= W-2 within
    `window` days and applied to days W .. W+6. Returns the causal out-of-sample prediction."""
    from sklearn.linear_model import Ridge
    ld = local_day.values.astype("datetime64[D]")
    days = np.unique(ld)
    pred = np.full(len(X), np.nan)
    ok_y = y.notna().values
    for b in range(0, len(days), refit):
        W = days[b]
        tr = ok_y & (ld <= W - np.timedelta64(2, "D")) & (ld > W - np.timedelta64(window + 2, "D"))
        if tr.sum() < 24 * 60:
            continue
        sw = None
        if half_life:
            age = (W - ld[tr]) / np.timedelta64(1, "D")
            sw = 0.5 ** (age / half_life)
        m = Ridge(alpha=1e-3).fit(X.values[tr], y.values[tr], sample_weight=sw)
        blk = (ld >= W) & (ld < W + np.timedelta64(refit, "D"))
        pred[blk] = m.predict(X.values[blk])
    return pd.Series(pred, index=X.index).clip(lower=0)


@lru_cache(maxsize=1)
def renewable_proxies():
    """Hourly UTC wind and solar estimates (MW) for every hour with weather forecasts."""
    w, s = load_weather(), load_smard()
    local_day = pd.Series(w.index.tz_localize("UTC").tz_convert(TZ).tz_localize(None).normalize().values,
                          index=w.index)
    wind_y = (s["wind_on"] + s["wind_off"]).reindex(w.index)
    solar_y = s["solar"].reindex(w.index)
    Xw = _wind_design(w)
    Xs = _solar_design(w)
    wind = _rolling_fit(Xw, wind_y, local_day, window=365, half_life=180)
    solar = _rolling_fit(Xs, solar_y, local_day, window=120, half_life=45)
    solar = solar.where(Xs["clear"].values > 0, 0.0)
    return wind.rename("wind_da"), solar.rename("solar_da")


def load_wind_solar():
    """(wind, solar) hourly UTC series in MW according to DE_RENEW."""
    if RENEW == "tso":
        s = load_smard()
        return (s["wind_on_fc"] + s["wind_off_fc"]).dropna(), s["solar_fc"].dropna()
    return renewable_proxies()


if __name__ == "__main__":
    s = load_smard()
    wind, solar = renewable_proxies()
    for name, est, tso, act in (("wind", wind, s["wind_on_fc"] + s["wind_off_fc"], s["wind_on"] + s["wind_off"]),
                                ("solar", solar, s["solar_fc"], s["solar"])):
        df = pd.concat([est, tso, act], axis=1, keys=["proxy", "tso", "actual"]).dropna()
        df = df[df.index >= "2024-01-01"]
        print(f"{name:5s} 2024+: MAE proxy {np.abs(df.proxy - df.actual).mean():6.0f} MW, "
              f"TSO forecast {np.abs(df.tso - df.actual).mean():6.0f} MW, mean actual {df.actual.mean():6.0f} MW")
