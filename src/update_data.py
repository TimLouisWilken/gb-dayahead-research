"""Download / refresh all raw inputs up to today from their public sources.

  prices    Ember European wholesale price data, hourly (GB = EMR/ENTSO-E day-ahead)
  gas       ONS release of National Gas System Average Price (SAP), daily, p/kWh
  demand    NESO historic demand data (national demand outturn, half-hourly)
  windfor   Elexon WINDFOR transmission wind forecast, issue before 07:30 UTC on D-1
  embedded  NESO embedded wind & solar forecasts, issue before 07:30 UTC on D-1
  ndf       Elexon NDF national demand forecast, latest issue by 09:00 UK time on D-1
  weather   Open-Meteo Previous Runs API, temperature forecast made 48 h before valid time
            (BMRS DGWS wind/solar forecasts are NOT used: they are published ~16:45 UTC on D-1,
            after the day-ahead auction)
  carbon    UK ETS (UKA) futures: no free API. Export "UK Emissions Allowances Energy c1
            Historical Data" (daily) from www.investing.com into data/manual/.
  ttf       ICE Endex Dutch TTF front-month settlement via Yahoo Finance, daily EUR/MWh
  windout   Elexon AGWS actual wind generation (transmission), half-hourly; published ~90 min
            after each half hour -> used only lagged >= 2 days (wind forecast-error features)

Usage: python src/update_data.py [prices gas demand renew weather carbon]
"""
import io
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW, MANUAL = ROOT / "data" / "raw", ROOT / "data" / "manual"
START = "2021-01-01"
TODAY = pd.Timestamp.today().normalize()
UA = {"User-Agent": "day-ahead-research/1.0"}


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


def parse_date(s):
    """ISO dates first (dayfirst would swap 2026-01-10 into 1 Oct), then day-first formats like 01-JAN-2021."""
    iso = pd.to_datetime(s, format="ISO8601", errors="coerce")
    return iso.fillna(pd.to_datetime(s[iso.isna()], format="mixed", dayfirst=True)).dt.date


# ------------------------------------------------------------------ prices
NEIGHBOUR_COUNTRIES = ["France", "Netherlands", "Belgium", "Norway", "Ireland", "Denmark", "Germany"]


def update_prices():
    """GB prices plus the day-ahead prices of the markets coupled to GB (one zip download)."""
    url = "https://files.ember-energy.org/public-downloads/price/outputs/european_wholesale_electricity_price_data_hourly.zip"
    z = zipfile.ZipFile(io.BytesIO(get(url).content))
    df = pd.read_csv(z.open("United Kingdom.csv"))
    df = df[["Datetime (UTC)", "Price (EUR/MWhe)"]]
    df.to_csv(RAW / "elec_price_hourly_ember.csv", index=False)
    print(f"prices : {df['Datetime (UTC)'].iloc[0]} -> {df['Datetime (UTC)'].iloc[-1]} ({len(df)} h)")
    parts = []
    for c in NEIGHBOUR_COUNTRIES:
        n = pd.read_csv(z.open(f"{c}.csv"), usecols=["Country", "Datetime (UTC)", "Price (EUR/MWhe)"])
        parts.append(n[n["Datetime (UTC)"] >= "2015-06-01"])
    out = pd.concat(parts).rename(columns={"Country": "country", "Datetime (UTC)": "datetime_utc",
                                           "Price (EUR/MWhe)": "price"})
    out.to_csv(RAW / "elec_price_hourly_neighbours.csv", index=False)
    print(f"neighbours: {len(NEIGHBOUR_COUNTRIES)} markets -> {out['datetime_utc'].max()}")


# ------------------------------------------------------------------ gas
def update_gas():
    page = get("https://www.ons.gov.uk/economy/economicoutputandproductivity/output/datasets/systemaveragepricesapofgas").text
    links = set(pd.Series(page.split('"')).loc[lambda s: s.str.contains(r"systemaveragepricesapofgas/20\d\d/.*\.xlsx")])
    link = max(links, key=lambda u: (u.split("systemaveragepricesapofgas/")[1][:4], u[-11:-5][4:] + u[-11:-5][2:4] + u[-11:-5][:2]))
    xl = pd.read_excel(io.BytesIO(get(link if link.startswith("http") else "https://www.ons.gov.uk" + link).content), sheet_name="1.Daily SAP Gas", header=4)
    df = xl[["Date", "SAP actual day"]].dropna()
    df = df[pd.to_numeric(df["SAP actual day"], errors="coerce").notna()]
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df.to_csv(RAW / "gas_sap_daily_ons.csv", index=False)
    print(f"gas    : {df.Date.iloc[0]} -> {df.Date.iloc[-1]} ({len(df)} d, p/kWh)  [{link.rsplit('/', 1)[-1]}]")


# ------------------------------------------------------------------ demand
def update_demand():
    meta = get("https://api.neso.energy/api/3/action/package_show", params={"id": "historic-demand-data"}).json()
    frames = []
    for res in meta["result"]["resources"]:
        year = "".join(ch for ch in res["name"] if ch.isdigit())
        if res["url"].endswith(".csv") and year.isdigit() and int(year) >= int(START[:4]):
            d = pd.read_csv(io.StringIO(get(res["url"]).text))
            d.columns = [c.upper() for c in d.columns]
            if "FORECAST_ACTUAL_INDICATOR" in d:
                d = d[d["FORECAST_ACTUAL_INDICATOR"] == "A"]
            d["SETTLEMENT_DATE"] = parse_date(d["SETTLEMENT_DATE"])
            frames.append(d[["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"]])
    df = pd.concat(frames).drop_duplicates(["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"], keep="last") \
        .sort_values(["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"])
    df.to_csv(RAW / "elec_demand_outturn_hh_neso.csv", index=False)
    print(f"demand : {df.SETTLEMENT_DATE.iloc[0]} -> {df.SETTLEMENT_DATE.iloc[-1]} ({len(df)} half-hours)")


# ------------------------------------------------------------------ forecast issue rule
# The GB day-ahead auctions for local delivery day D close at ~09:20 UK time on D-1.
# Every forecast input must have been *issued* by 07:30 UTC on D-1 (= 07:30 GMT / 08:30 BST).
ISSUE_CUTOFF = pd.Timedelta(hours=7, minutes=30)
TZ = "Europe/London"


def local_day(t_utc):
    """Local (UK) delivery date of UTC timestamps, as tz-naive midnight timestamps."""
    return pd.DatetimeIndex(t_utc).tz_convert(TZ).tz_localize(None).normalize()


# ------------------------------------------------------------------ transmission wind (WINDFOR)
def update_windfor():
    """Elexon WINDFOR (NESO transmission-connected wind forecast), the issue published before
    07:30 UTC on D-1 for every local delivery day D."""
    from concurrent.futures import ThreadPoolExecutor
    path = RAW / "wind_forecast_windfor_d1_morning.csv"
    old = pd.read_csv(path, parse_dates=["target_utc", "publish_utc"]) if path.exists() else None
    days = pd.date_range(START, TODAY + pd.Timedelta(days=1), freq="D")
    if old is not None:
        days = days[days > old["delivery_day"].max()]
    url = "https://data.elexon.co.uk/bmrs/api/v1/forecast/generation/wind/history"

    def fetch(d):
        cutoff = d - pd.Timedelta(days=1) + ISSUE_CUTOFF
        js = get(url, params={"publishTime": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")}).json().get("data", [])
        if not js:
            return None
        f = pd.DataFrame(js)
        f["target_utc"] = pd.to_datetime(f["startTime"], utc=True)
        f["publish_utc"] = pd.to_datetime(f["publishTime"], utc=True)
        f = f[(local_day(f["target_utc"]) == d) & (f["publish_utc"] <= cutoff.tz_localize("UTC"))]
        f["delivery_day"] = d.date()
        return f[["delivery_day", "target_utc", "publish_utc", "generation"]]

    with ThreadPoolExecutor(6) as ex:
        parts = [p for p in ex.map(fetch, days) if p is not None and len(p)]
    new = pd.concat(parts) if parts else pd.DataFrame()
    for c in ("target_utc", "publish_utc"):
        if len(new):
            new[c] = new[c].dt.tz_convert(None)
    df = pd.concat([old, new]) if old is not None else new
    df["delivery_day"] = pd.to_datetime(df["delivery_day"]).dt.date
    df = df.drop_duplicates("target_utc", keep="last").sort_values("target_utc")
    df.to_csv(path, index=False)
    lag = (pd.to_datetime(df["delivery_day"]) - pd.to_datetime(df["publish_utc"])).dt.total_seconds() / 3600
    print(f"windfor: {df.delivery_day.min()} -> {df.delivery_day.max()} ({len(df)} h), "
          f"issued {lag.min():.1f}-{lag.max():.1f} h before local midnight of D")


# ------------------------------------------------------------------ embedded wind & solar (NESO)
def update_embedded():
    """NESO embedded wind & solar forecasts: for every half-hour of local day D keep the latest
    forecast issued before 07:30 UTC on D-1. Yearly archives are streamed (~0.5 GB each)."""
    meta = get("https://api.neso.energy/api/3/action/package_show", params={"id": "embedded-wind-and-solar-forecasts"})
    res = {r["name"].split()[-1]: r["url"] for r in meta.json()["result"]["resources"] if "Archive" in r["name"]}
    path = RAW / "embedded_forecast_d1_morning.csv"
    old = pd.read_csv(path, parse_dates=["target_utc", "issue_utc"]) if path.exists() else None
    years = [y for y in sorted(res) if int(y) >= int(START[:4])]
    if old is not None:     # only re-read the current year (and the last one, for late updates)
        years = [y for y in years if int(y) >= old["target_utc"].max().year - 1]
    keep = []
    cache = Path.home() / ".cache" / "day-ahead-prediction"   # ~3 GB, kept out of iCloud
    cache.mkdir(parents=True, exist_ok=True)
    for y in years:
        f = cache / f"embedded_archive_{y}.csv"
        if not f.exists() or y == str(TODAY.year):
            # curl: follows the storage redirect, resumes broken transfers, decompresses gzip
            tmp = f.with_suffix(".part")
            subprocess.run(["curl", "-sSL", "--compressed", "--retry", "5", "-C", "-", "-o", str(tmp), res[y]],
                           check=True)
            tmp.rename(f)
        n = 0
        for ch in pd.read_csv(f, chunksize=1_000_000,
                              usecols=["DATE_GMT", "TIME_GMT", "EMBEDDED_WIND_FORECAST",
                                       "EMBEDDED_SOLAR_FORECAST", "Forecast_Datetime"]):
            n += len(ch)
            end = pd.to_datetime(ch["DATE_GMT"].str[:10] + " " + ch["TIME_GMT"].str[:5], format="%Y-%m-%d %H:%M")
            tgt = end - pd.Timedelta(minutes=30)                 # TIME_GMT is the end of the half-hour
            issue = pd.to_datetime(ch["Forecast_Datetime"].str[:19], format="%Y-%m-%dT%H:%M:%S")
            day = local_day(tgt.dt.tz_localize("UTC"))
            cutoff = day - pd.Timedelta(days=1) + ISSUE_CUTOFF
            ok = (issue.values <= cutoff.values) & (issue.values >= (cutoff - pd.Timedelta(hours=8)).values)
            if ok.any():
                keep.append(pd.DataFrame({"target_utc": tgt[ok].values, "issue_utc": issue[ok].values,
                                          "emb_wind": ch["EMBEDDED_WIND_FORECAST"].values[ok],
                                          "emb_solar": ch["EMBEDDED_SOLAR_FORECAST"].values[ok]}))
        print(f"embedded: {y} scanned {n:,} rows", flush=True)
    new = pd.concat(keep)
    new = new.sort_values("issue_utc").drop_duplicates("target_utc", keep="last")   # latest issue before cutoff
    df = pd.concat([old, new]) if old is not None else new
    df = df.drop_duplicates("target_utc", keep="last").sort_values("target_utc")
    df.to_csv(path, index=False)
    print(f"embedded: {df.target_utc.min()} -> {df.target_utc.max()} ({len(df)} half-hours)")


# ------------------------------------------------------------------ demand forecast (NDF)
# NESO publishes the half-hourly National Demand Forecast (Elexon dataset NDF) every 30 min.
# The issue covering the whole next "electricity day" (05:00-05:00 local) appears at 08:45 UK
# local time, i.e. before both GB day-ahead gate closures (EPEX 09:20, N2EX 09:50 UK time).
# Cutoff for this input: 09:00 UK local time on D-1.
NDF_CUTOFF_LOCAL = pd.Timedelta(hours=9)


def update_ndf():
    """For every half-hour of local day D: the latest NDF issue published by 09:00 UK time on D-1."""
    from concurrent.futures import ThreadPoolExecutor
    path = RAW / "demand_forecast_ndf_d1.csv"
    old = pd.read_csv(path, parse_dates=["target_utc", "publish_utc"]) if path.exists() else None
    days = pd.date_range(START, TODAY + pd.Timedelta(days=1), freq="D")
    if old is not None:
        days = days[days > pd.Timestamp(old["delivery_day"].max())]
    url = "https://data.elexon.co.uk/bmrs/api/v1/datasets/NDF"

    def fetch(d):
        cutoff = (d - pd.Timedelta(days=1) + NDF_CUTOFF_LOCAL).tz_localize(TZ).tz_convert("UTC")
        js = get(url, params={"publishDateTimeFrom": (cutoff - pd.Timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                              "publishDateTimeTo": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")}).json().get("data", [])
        if not js:
            return None
        f = pd.DataFrame(js)
        f["target_utc"] = pd.to_datetime(f["startTime"], utc=True)
        f["publish_utc"] = pd.to_datetime(f["publishTime"], utc=True)
        f = f[(local_day(f["target_utc"]) == d) & (f["publish_utc"] <= cutoff)]
        f = f.sort_values("publish_utc").drop_duplicates("target_utc", keep="last")   # latest issue per half-hour
        f["delivery_day"] = d.date()
        return f[["delivery_day", "target_utc", "publish_utc", "demand"]]

    with ThreadPoolExecutor(6) as ex:
        parts = [p for p in ex.map(fetch, days) if p is not None and len(p)]
    new = pd.concat(parts) if parts else pd.DataFrame()
    for c in ("target_utc", "publish_utc"):
        if len(new):
            new[c] = new[c].dt.tz_convert(None)
    df = pd.concat([old, new]) if old is not None else new
    df["delivery_day"] = pd.to_datetime(df["delivery_day"]).dt.date
    df = df.drop_duplicates("target_utc", keep="last").sort_values("target_utc")
    df.to_csv(path, index=False)
    per_day = df.groupby("delivery_day").size()
    print(f"ndf    : {df.delivery_day.min()} -> {df.delivery_day.max()} ({len(df)} half-hours, "
          f"{(per_day >= 46).mean():.1%} of days complete)")


# ------------------------------------------------------------------ temperature forecast
# previous_day2 = value forecast 48 h before the valid time, so even the last hour of day D was
# forecast before D-1 00:00 UTC. (previous_day1 would use runs issued during D-1, after the auction.)
# JMA is the only model whose previous-day archive reaches back to 2021 without gaps.
WEATHER_MODEL, WEATHER_VAR = "jma_seamless", "temperature_2m_previous_day2"
DEMAND_POINT = (52.48, -1.90)   # Birmingham, roughly the population centre of GB


def update_weather():
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    parts = []
    for y0 in range(int(START[:4]), TODAY.year + 1):
        s, e = pd.Timestamp(f"{y0}-01-01"), min(pd.Timestamp(f"{y0}-12-31"), TODAY + pd.Timedelta(days=1))
        js = get(url, params={"latitude": DEMAND_POINT[0], "longitude": DEMAND_POINT[1], "start_date": s.date(),
                              "end_date": e.date(), "hourly": WEATHER_VAR, "timezone": "GMT",
                              "models": WEATHER_MODEL}).json()
        parts.append(pd.DataFrame(js["hourly"]))
        time.sleep(1)
    df = pd.concat(parts).rename(columns={"time": "datetime_utc", WEATHER_VAR: "temperature"})
    df.to_csv(RAW / "temperature_forecast_d2_openmeteo.csv", index=False)
    ok = df.dropna()
    print(f"weather: {WEATHER_MODEL} {WEATHER_VAR} {ok.datetime_utc.iloc[0]} -> {ok.datetime_utc.iloc[-1]}, "
          f"missing {df.temperature.isna().mean():.3%}")


def update_margin():
    """NESO national surplus forecast (Elexon OCNMFD, 'OC2' 2-14 day margin, MW). The issues of
    day D-2 are the freshest that already contain delivery day D (the 2-14 day window of D-1
    issues starts at D+1), and all of D-2 lies safely before the 07:30 UTC D-1 cutoff.
    Used by the spike classifier / probabilistic layer; history starts mid-2021."""
    from concurrent.futures import ThreadPoolExecutor
    path = RAW / "margin_forecast_ocnmfd.csv"
    old = pd.read_csv(path, parse_dates=["publish_utc"]) if path.exists() else None
    days = pd.date_range("2021-07-01", TODAY + pd.Timedelta(days=1), freq="D")
    if old is not None:
        days = days[days > pd.Timestamp(old["delivery_day"].max())]
    url = "https://data.elexon.co.uk/bmrs/api/v1/datasets/OCNMFD"

    def fetch(d):
        pub = d - pd.Timedelta(days=2)
        js = get(url, params={"publishDateTimeFrom": pub.strftime("%Y-%m-%dT00:00:00Z"),
                              "publishDateTimeTo": pub.strftime("%Y-%m-%dT23:59:59Z"),
                              "format": "json"}).json().get("data", [])
        f = pd.DataFrame(js)
        if not len(f):
            return None
        f = f[f["forecastDate"] == d.strftime("%Y-%m-%d")].sort_values("publishTime")
        if not len(f):
            return None
        return {"delivery_day": d.date(), "publish_utc": f["publishTime"].iloc[-1],
                "surplus": float(f["surplus"].iloc[-1])}

    with ThreadPoolExecutor(6) as ex:
        rows = [r for r in ex.map(fetch, days) if r is not None]
    new = pd.DataFrame(rows)
    if len(new):
        new["publish_utc"] = pd.to_datetime(new["publish_utc"]).dt.tz_convert(None)
    df = pd.concat([old, new]) if old is not None else new
    df["delivery_day"] = pd.to_datetime(df["delivery_day"]).dt.date
    df = df.drop_duplicates("delivery_day", keep="last").sort_values("delivery_day")
    df.to_csv(path, index=False)
    print(f"margin : OCNMFD {df.delivery_day.min()} -> {df.delivery_day.max()} ({len(df)} days)")


def update_remit():
    """Elexon REMIT unavailability messages (planned & unplanned outages of generation units
    AND interconnectors, with publish time, event window, revision chain and status). The only
    forward-looking scarcity information for delivery day D that is complete at the 07:30 UTC
    D-1 cutoff. Backfilled from 2019-06 so that long-lead outages announced years ahead are
    known for the 2021-07+ model period. Stored compressed outside iCloud (~500k messages)."""
    cache = Path.home() / ".cache" / "day-ahead-prediction"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "remit_messages.csv.gz"
    cols = ["mrid", "revisionNumber", "publishTime", "eventStartTime", "eventEndTime",
            "eventStatus", "eventType", "assetId", "fuelType", "normalCapacity",
            "unavailableCapacity", "unavailabilityType"]
    old = pd.read_csv(path, parse_dates=["publishTime"]) if path.exists() else None
    start = pd.Timestamp("2019-06-01")
    if old is not None and len(old):
        start = old["publishTime"].max().tz_localize(None).normalize()  # refetch last day (revisions)
    url = "https://data.elexon.co.uk/bmrs/api/v1/datasets/REMIT/stream"
    frames = []
    t = start
    while t <= TODAY:
        t2 = t + pd.Timedelta(days=13)
        js = get(url, params={"publishDateTimeFrom": t.strftime("%Y-%m-%dT00:00:00Z"),
                              "publishDateTimeTo": t2.strftime("%Y-%m-%dT23:59:59Z")}).json()
        if js:
            frames.append(pd.DataFrame(js)[[c for c in cols if c in js[0]]])
        t = t2 + pd.Timedelta(days=1)
    new = pd.concat(frames) if frames else pd.DataFrame(columns=cols)
    if len(new):
        new["publishTime"] = pd.to_datetime(new["publishTime"], format="ISO8601", utc=True).dt.tz_convert(None)
    df = pd.concat([old, new]) if old is not None else new
    df = (df.sort_values("publishTime")
            .drop_duplicates(["mrid", "revisionNumber", "publishTime"], keep="last"))
    df.to_csv(path, index=False)
    print(f"remit  : {df.publishTime.min()} -> {df.publishTime.max()} ({len(df)} messages)")


# population-weighted temperature (EXTRA_FEATS=4): four cities roughly covering the GB
# population (South incl. London ~45 %, North England ~25 %, Midlands/Wales ~20 %, Scotland ~10 %)
CITIES = {"london": (51.51, -0.13), "manchester": (53.48, -2.24),
          "birmingham": (52.48, -1.90), "glasgow": (55.86, -4.25)}


def update_weather_multi():
    """Temperature forecasts for four cities, same previous_day2 (48 h lead) rule as
    update_weather(). Used by the population-weighted temperature features (EXTRA_FEATS=4)."""
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    out = None
    for name, (lat, lon) in CITIES.items():
        parts = []
        for y0 in range(int(START[:4]), TODAY.year + 1):
            s, e = pd.Timestamp(f"{y0}-01-01"), min(pd.Timestamp(f"{y0}-12-31"), TODAY + pd.Timedelta(days=1))
            js = get(url, params={"latitude": lat, "longitude": lon, "start_date": s.date(),
                                  "end_date": e.date(), "hourly": WEATHER_VAR, "timezone": "GMT",
                                  "models": WEATHER_MODEL}).json()
            parts.append(pd.DataFrame(js["hourly"]))
            time.sleep(1)
        s = pd.concat(parts).rename(columns={"time": "datetime_utc", WEATHER_VAR: name}).set_index("datetime_utc")
        out = s if out is None else out.join(s, how="outer")
        print(f"weather_multi: {name} done", flush=True)
    out.to_csv(RAW / "temperature_forecast_d2_multicity.csv")
    print(f"weather_multi: {out.index.min()} -> {out.index.max()}, missing {out.isna().mean().mean():.3%}")


# ------------------------------------------------------------------ carbon
def update_carbon():
    """UK ETS allowance (UKA) futures, front month, GBP/t, from investing.com CSV exports
    ("UK Emissions Allowances Energy c1 Historical Data") placed in data/manual/.

    Note: the UCL file carbon_ukets_futures_daily_investing.csv is in fact the *EU* ETS (EUA)
    price despite its name: it starts at 33.89 on 4 Jan 2021, before UKA futures existed
    (May 2021), and runs ~1.46x above UKA in Dec 2025. It is kept only as a fallback."""
    files = sorted(MANUAL.glob("*.csv"))
    if not files:
        print("carbon : no investing.com UKA export in data/manual/ -> nothing to do")
        return
    frames = []
    for f in files:
        d = pd.read_csv(f, encoding="utf-8-sig")
        # English investing.com exports use MM/DD/YYYY; fall back to DD/MM/YYYY
        dt = pd.to_datetime(d["Date"], format="%m/%d/%Y", errors="coerce")
        if dt.isna().any():
            dt = pd.to_datetime(d["Date"], format="%d/%m/%Y")
        price = pd.to_numeric(d["Price"].astype(str).str.replace(",", ""), errors="coerce")
        frames.append(pd.DataFrame({"Date": dt.dt.date, "Price": price}))
    df = pd.concat(frames).dropna().drop_duplicates("Date", keep="last").sort_values("Date")
    df.to_csv(RAW / "carbon_uka_daily_investing.csv", index=False)
    print(f"carbon : UKA {df.Date.iloc[0]} -> {df.Date.iloc[-1]} ({len(df)} trading days, GBP/t) from {len(files)} file(s)")


# ------------------------------------------------------------------ TTF gas futures
def update_ttf():
    """ICE Endex Dutch TTF front-month settlement (EUR/MWh) via Yahoo Finance. The settlement of
    trading day t is published the same evening, so for delivery day D the settlement of D-2 is
    safely known at the decision time (07:30 UTC on D-1)."""
    import yfinance as yf
    df = yf.download("TTF=F", start=START, progress=False, auto_adjust=True)["Close"]
    s = df.iloc[:, 0].rename("ttf").dropna()
    s.to_csv(RAW / "gas_ttf_daily_yahoo.csv", index_label="Date")
    print(f"ttf    : {s.index.min().date()} -> {s.index.max().date()} ({len(s)} trading days, EUR/MWh)")


# ------------------------------------------------------------------ wind outturn (AGWS)
def update_windout():
    """Elexon AGWS actual wind generation (transmission connected, onshore + offshore), half-hourly.
    Published ~90 minutes after each half hour, so the full day D-2 is known at 07:30 UTC on D-1.
    Used only for the lagged wind forecast-error features (never the delivery day itself)."""
    from concurrent.futures import ThreadPoolExecutor
    path = RAW / "wind_outturn_hh_elexon.csv"
    old = pd.read_csv(path, parse_dates=["start_utc", "publish_utc"]) if path.exists() else None
    start = old["start_utc"].max() + pd.Timedelta(minutes=30) if old is not None else pd.Timestamp(START)
    url = "https://data.elexon.co.uk/bmrs/api/v1/generation/actual/per-type/wind-and-solar"
    windows = pd.date_range(start.floor("D"), TODAY + pd.Timedelta(days=1), freq="6D")

    def fetch(t0):
        t1 = t0 + pd.Timedelta(days=6)
        js = get(url, params={"from": t0.strftime("%Y-%m-%dT%H:%MZ"), "to": t1.strftime("%Y-%m-%dT%H:%MZ"),
                              "format": "json"}).json().get("data", [])
        f = pd.DataFrame(js)
        if not len(f):
            return None
        f = f[f["businessType"] == "Wind generation"]
        # a period can be published several times (revisions). The value of local day X is used
        # for delivery days >= X+2, decided at 07:30 UTC on X+1: keep only publications before
        # that cutoff, and of those the latest per (period, psrType).
        start = pd.to_datetime(f["startTime"], utc=True)
        day = start.dt.tz_convert("Europe/London").dt.tz_localize(None).dt.normalize()
        cutoff = day + pd.Timedelta(days=1, hours=7, minutes=30)
        f = f[pd.to_datetime(f["publishTime"], utc=True).dt.tz_localize(None).values <= cutoff.values]
        f = f.sort_values("publishTime").drop_duplicates(["startTime", "psrType"], keep="last")
        g = f.groupby("startTime").agg(wind=("quantity", "sum"), publish=("publishTime", "max")).reset_index()
        return g

    with ThreadPoolExecutor(6) as ex:
        parts = [p for p in ex.map(fetch, windows) if p is not None]
    new = pd.concat(parts).rename(columns={"startTime": "start_utc", "publish": "publish_utc"})
    for c in ("start_utc", "publish_utc"):
        new[c] = pd.to_datetime(new[c], utc=True).dt.tz_convert(None)
    new = new[new["start_utc"] >= start]
    df = pd.concat([old, new]) if old is not None else new
    df = df.drop_duplicates("start_utc").sort_values("start_utc")
    df.to_csv(path, index=False, columns=["start_utc", "wind", "publish_utc"])
    print(f"windout: {df['start_utc'].min()} -> {df['start_utc'].max()} ({len(df)} half hours)")


# ------------------------------------------------------------------ live top-ups (src/live.py)
# The bulk sources above lag reality: Ember prices by days-weeks, the NESO demand file by weeks,
# the embedded archive download is 0.5 GB. These three steps close the gap to "now" from live
# APIs so that live.py can forecast tomorrow. data.py merges them after the end of the bulk file.
def update_prices_live():
    """N2EX day-ahead auction results (hourly, EUR/MWh) from the Nord Pool data portal. The price
    of local day D is published on D-1 (~10:45 UK time). Overlap check vs Ember: mean diff
    0.05 EUR, max < 4 EUR. Used by data.load_prices() only after the last Ember timestamp."""
    path = RAW / "elec_price_hourly_nordpool.csv"
    old = pd.read_csv(path, parse_dates=["datetime_utc"]) if path.exists() else None
    ember_end = pd.Timestamp(pd.read_csv(RAW / "elec_price_hourly_ember.csv")["Datetime (UTC)"].iloc[-1])
    start = (old["datetime_utc"].max() if old is not None else ember_end).normalize() - pd.Timedelta(days=2)
    rows = []
    for d in pd.date_range(start, TODAY + pd.Timedelta(days=1), freq="D"):
        r = get("https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices",
                params={"date": d.strftime("%Y-%m-%d"), "market": "N2EX_DayAhead",
                        "deliveryArea": "UK", "currency": "EUR"})
        if r.status_code == 204 or not r.content:       # not auctioned yet
            continue
        for e in r.json().get("multiAreaEntries", []):
            if e["entryPerArea"].get("UK") is not None:
                rows.append((pd.Timestamp(e["deliveryStart"]).tz_convert(None), float(e["entryPerArea"]["UK"])))
    new = pd.DataFrame(rows, columns=["datetime_utc", "price"])
    df = pd.concat([old, new]) if old is not None else new
    df = df.drop_duplicates("datetime_utc", keep="last").sort_values("datetime_utc")
    df.to_csv(path, index=False)
    print(f"prices_live : N2EX/Nordpool {df.datetime_utc.min()} -> {df.datetime_utc.max()} "
          f"({len(df)} h; Ember ends {ember_end})")


def update_demand_live():
    """Elexon INDO initial national demand outturn (half-hourly), extending the NESO ND file.
    INDO is near-identical to NESO ND (mean diff ~7 MW on ~24 GW); published ~30 min after each
    settlement period, and the features only use demand lagged >= 2 days."""
    path = RAW / "elec_demand_outturn_hh_indo.csv"
    old = pd.read_csv(path, parse_dates=["start_utc"]) if path.exists() else None
    neso_end = pd.Timestamp(pd.read_csv(RAW / "elec_demand_outturn_hh_neso.csv").iloc[-1, 0])
    start = (old["start_utc"].max().normalize() if old is not None else neso_end) - pd.Timedelta(days=2)
    frames = []
    for t0 in pd.date_range(start, TODAY, freq="6D"):
        js = get("https://data.elexon.co.uk/bmrs/api/v1/demand/outturn",
                 params={"settlementDateFrom": t0.strftime("%Y-%m-%d"),
                         "settlementDateTo": min(t0 + pd.Timedelta(days=5), TODAY).strftime("%Y-%m-%d"),
                         "format": "json"}).json().get("data", [])
        if js:
            f = pd.DataFrame(js)
            frames.append(pd.DataFrame({"start_utc": pd.to_datetime(f["startTime"], utc=True).dt.tz_convert(None),
                                        "demand": pd.to_numeric(f["initialDemandOutturn"])}))
    new = pd.concat(frames)
    df = pd.concat([old, new]) if old is not None else new
    df = df.drop_duplicates("start_utc", keep="last").sort_values("start_utc")
    df.to_csv(path, index=False)
    print(f"demand_live : INDO {df.start_utc.min()} -> {df.start_utc.max()} "
          f"({len(df)} half-hours; NESO file ends {neso_end.date()})")


def update_embedded_live():
    """Latest embedded wind & solar forecast issues from the NESO datastore (SQL API) - a light
    daily top-up of embedded_forecast_d1_morning.csv applying exactly the same issue rule as the
    archive scan in update_embedded(): latest issue <= 07:30 UTC on D-1 per half hour of D."""
    path = RAW / "embedded_forecast_d1_morning.csv"
    old = pd.read_csv(path, parse_dates=["target_utc", "issue_utc"])
    t0 = old["target_utc"].max().normalize() - pd.Timedelta(days=2)
    meta = get("https://api.neso.energy/api/3/action/package_show",
               params={"id": "embedded-wind-and-solar-forecasts"}).json()
    res = {r["name"].split()[-1]: r["url"].rsplit("/", 1)[-1] for r in meta["result"]["resources"]
           if "Archive" in r["name"]}
    keep = []
    for y in sorted({t0.year, (TODAY + pd.Timedelta(days=1)).year}):
        sql = ('SELECT "DATE_GMT","TIME_GMT","EMBEDDED_WIND_FORECAST","EMBEDDED_SOLAR_FORECAST","Forecast_Datetime" '
               f'FROM "{res[str(y)]}" WHERE "DATE_GMT" >= \'{t0.strftime("%Y-%m-%d")}\'')
        js = get("https://api.neso.energy/api/3/action/datastore_search_sql", params={"sql": sql}).json()
        ch = pd.DataFrame(js["result"]["records"])
        if not len(ch):
            continue
        end = pd.to_datetime(ch["DATE_GMT"].str[:10] + " " + ch["TIME_GMT"].str[:5], format="%Y-%m-%d %H:%M")
        tgt = end - pd.Timedelta(minutes=30)                 # TIME_GMT is the end of the half-hour
        issue = pd.to_datetime(ch["Forecast_Datetime"].str[:19], format="%Y-%m-%dT%H:%M:%S")
        day = local_day(tgt.dt.tz_localize("UTC"))
        cutoff = day - pd.Timedelta(days=1) + ISSUE_CUTOFF
        ok = (issue.values <= cutoff.values) & (issue.values >= (cutoff - pd.Timedelta(hours=8)).values)
        if ok.any():
            keep.append(pd.DataFrame({"target_utc": tgt[ok].values, "issue_utc": issue[ok].values,
                                      "emb_wind": pd.to_numeric(ch["EMBEDDED_WIND_FORECAST"]).values[ok],
                                      "emb_solar": pd.to_numeric(ch["EMBEDDED_SOLAR_FORECAST"]).values[ok]}))
    new = pd.concat(keep).sort_values("issue_utc").drop_duplicates("target_utc", keep="last")
    df = pd.concat([old, new]).drop_duplicates("target_utc", keep="last").sort_values("target_utc")
    df.to_csv(path, index=False)
    print(f"embedded_live: {df.target_utc.min()} -> {df.target_utc.max()} ({len(df)} half-hours)")


STEPS = {"prices": update_prices, "gas": update_gas, "demand": update_demand, "windfor": update_windfor, "ndf": update_ndf,
         "embedded": update_embedded, "weather": update_weather, "carbon": update_carbon,
         "ttf": update_ttf, "windout": update_windout,
         "prices_live": update_prices_live, "demand_live": update_demand_live, "embedded_live": update_embedded_live,
         "weather_multi": update_weather_multi, "margin": update_margin, "remit": update_remit}

if __name__ == "__main__":
    for step in sys.argv[1:] or STEPS:
        STEPS[step]()
