"""Download / refresh the raw inputs for the German (DE-LU) day-ahead market into data/raw/de/.

  smard     SMARD.de (Bundesnetzagentur, CC BY 4.0), hourly, UTC:
              price      DE-LU day-ahead price, EUR/MWh (filter 4169). Since the switch of the
                         SDAC to 15-minute products (2025-10-01) the hourly value is the mean of the
                         four quarter-hour prices; before that it is the hourly auction price.
              load_fc    day-ahead total load forecast (411). ENTSO-E Transparency Regulation
                         (EU 543/2013, Art. 6): published at the latest two hours before the
                         day-ahead gate closure (12:00 CET), i.e. known at the decision time.
              wind_on_fc, wind_off_fc, solar_fc
                         day-ahead generation forecasts (123, 3791, 125). Legal publication
                         deadline is 18:00 on D-1 (Art. 14 / EEV), so the published vintage is not
                         guaranteed to precede the auction; the TSOs sell exactly these
                         forecast volumes in the auction, i.e. a forecast exists before 12:00.
                         This is the information set of the open EPF benchmark (Lago et al. 2021)
                         and is flagged as the one relaxation against the GB set-up.
              load       realised total load (410), used only with lags >= 2 days
              wind_on, wind_off, solar
                         realised wind and PV generation (4067, 1225, 4068), lags >= 2 days only
            The TSO wind/solar forecasts are stored for comparison only: on 2026-10-20 at 10:54
            CEST (one hour before gate closure) SMARD already had the load forecast for D but not
            the wind/solar forecasts, so they are not used in the strict information set.
  weather   Open-Meteo Previous Runs (JMA), forecasts made 48 h before each valid hour:
            temperature (population-weighted over six regional centres), 10 m wind speed at
            eight wind points, cloud cover at six PV points (inputs of the wind/solar proxies)
  ttf       ICE Endex Dutch TTF front-month settlement via Yahoo Finance, EUR/MWh
  eua       EU ETS allowance price, EUR/t: SparkChange Physical Carbon EUA ETC (LSE: CO2, one
            share = one EUA) from its listing in Oct 2021; before that WisdomTree Carbon (LSE:
            CARB, tracks EUA futures), rescaled by the median CO2/CARB ratio of the first 20
            common trading days (no later information)

Usage: MARKET is not needed here.  python src/update_data_de.py [smard weather ttf eua]
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "de"
START = "2018-10-01"                      # DE-LU bidding zone (split from Austria)
TODAY = pd.Timestamp.today().normalize()
UA = {"User-Agent": "day-ahead-research/1.0"}
SMARD = "https://www.smard.de/app/chart_data"
SMARD_SERIES = {                          # name: (filter, region)
    "price": (4169, "DE-LU"), "load_fc": (411, "DE"), "wind_on_fc": (123, "DE"),
    "wind_off_fc": (3791, "DE"), "solar_fc": (125, "DE"), "load": (410, "DE"),
    "wind_on": (4067, "DE"), "wind_off": (1225, "DE"), "solar": (4068, "DE"),
}
# Weather points for the strict wind/solar proxies (JMA, 48 h lead): onshore wind regions,
# two offshore clusters, and PV regions. JMA is the only model in the previous-runs archive
# that reaches back to 2018 without gaps; it provides 10 m wind and cloud cover, no radiation.
WIND_POINTS = {"sh": (54.3, 9.6), "nds": (52.9, 8.0), "bb": (52.4, 13.5), "st": (51.9, 11.8),
               "mv": (53.8, 12.5), "nrw": (51.6, 7.5), "off_north": (54.3, 6.5), "off_baltic": (54.7, 13.9)}
PV_POINTS = {"by": (48.6, 11.5), "bw": (48.7, 9.0), "nrw": (51.4, 7.3), "nds": (52.6, 9.5),
             "bb": (52.2, 13.6), "sn": (50.9, 12.9)}
# population-weighted temperature: regional centres and approximate population shares (2023)
CITIES = {"berlin": (52.52, 13.40, 0.19), "hamburg": (53.55, 9.99, 0.16), "cologne": (50.94, 6.96, 0.22),
          "frankfurt": (50.11, 8.68, 0.13), "stuttgart": (48.78, 9.18, 0.13), "munich": (48.14, 11.58, 0.17)}


def get(url, **kw):
    for attempt in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=120, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == 3:
                raise
            print(f"   retry after error: {e}")
            time.sleep(3 * (attempt + 1))


def smard_series(filt, region):
    """Full hourly history of one SMARD series as a UTC-indexed Series (weekly chunk files)."""
    ts = get(f"{SMARD}/{filt}/{region}/index_hour.json").json()["timestamps"]
    t0 = pd.Timestamp(START).value // 10**6 - 7 * 86400 * 1000
    parts = []
    for t in [x for x in ts if x >= t0]:
        s = get(f"{SMARD}/{filt}/{region}/{filt}_{region}_hour_{t}.json").json()["series"]
        parts.append(pd.DataFrame(s, columns=["t", "v"]))
    df = pd.concat(parts).dropna()
    s = pd.Series(df["v"].values, index=pd.to_datetime(df["t"], unit="ms")).sort_index()
    return s[~s.index.duplicated(keep="last")]


def update_smard():
    RAW.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, (f, reg) in SMARD_SERIES.items():
        out[name] = smard_series(f, reg)
        s = out[name]
        print(f"smard  : {name:12s} {s.index.min()} -> {s.index.max()} ({len(s)} hours)")
    df = pd.DataFrame(out)
    df.index.name = "datetime_utc"
    df.to_csv(RAW / "smard_hourly.csv")


def _jma(lat, lon, var):
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    parts = []
    for y0 in range(int(START[:4]), TODAY.year + 1):
        s, e = pd.Timestamp(f"{y0}-01-01"), min(pd.Timestamp(f"{y0}-12-31"), TODAY + pd.Timedelta(days=1))
        js = get(url, params={"latitude": lat, "longitude": lon, "start_date": s.date(), "end_date": e.date(),
                              "hourly": var, "timezone": "GMT", "models": "jma_seamless"}).json()
        parts.append(pd.DataFrame(js["hourly"]).set_index("time")[var])
        time.sleep(0.5)
    out = pd.concat(parts)
    out.index = pd.to_datetime(out.index, format="ISO8601")
    return out


def update_weather():
    t = pd.DataFrame({c: _jma(lat, lon, "temperature_2m_previous_day2") for c, (lat, lon, _) in CITIES.items()})
    t["temp_pop"] = sum(t[c] * w for c, (_, _, w) in CITIES.items()) / sum(w for *_, w in CITIES.values())
    w = pd.DataFrame({f"wind_{c}": _jma(*p, "wind_speed_10m_previous_day2") for c, p in WIND_POINTS.items()})
    cc = pd.DataFrame({f"cc_{c}": _jma(*p, "cloud_cover_previous_day2") for c, p in PV_POINTS.items()})
    df = pd.concat([t[["temp_pop"]], w, cc], axis=1)
    df.index.name = "datetime_utc"
    df.to_csv(RAW / "weather_d2_de.csv")
    print(f"weather: {df.index.min()} -> {df.index.max()}, missing {df.isna().mean().max():.3%} (worst column)")


def _yahoo(ticker, start):
    import yfinance as yf
    return yf.download(ticker, start=start, progress=False, auto_adjust=True)["Close"].iloc[:, 0].dropna()


def update_ttf():
    s = _yahoo("TTF=F", "2018-01-01").rename("ttf")
    s.to_csv(RAW / "gas_ttf_daily_yahoo.csv", index_label="Date")
    print(f"ttf    : {s.index.min().date()} -> {s.index.max().date()} ({len(s)} trading days, EUR/MWh)")


def update_eua():
    phys, carb = _yahoo("CO2.L", "2021-01-01"), _yahoo("CARB.L", "2018-01-01")
    ratio = (phys / carb).dropna().iloc[:20].median()       # first 20 common days only
    s = pd.concat([carb[carb.index < phys.index.min()] * ratio, phys]).rename("eua")
    s.to_csv(RAW / "carbon_eua_daily_yahoo.csv", index_label="Date")
    print(f"eua    : {s.index.min().date()} -> {s.index.max().date()} ({len(s)} trading days, EUR/t; "
          f"CARB->EUA ratio {ratio:.3f} before {phys.index.min().date()})")


STEPS = {"smard": update_smard, "weather": update_weather, "ttf": update_ttf, "eua": update_eua}

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for step in sys.argv[1:] or STEPS:
        STEPS[step]()
