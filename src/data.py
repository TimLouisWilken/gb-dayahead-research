"""Data loading and leakage-free feature construction for GB day-ahead price forecasting.

Forecasting set-up (Lago et al., 2021): before the GB day-ahead auctions close on day D-1
(~09:20 UK time) we forecast the 24 hourly prices of local delivery day D. Days are UK
local days, as in the auctions; DST days are mapped to 24 hours (spring: the missing hour
is interpolated within the day, autumn: the doubled hour is averaged), as in epftoolbox.

Information set at the decision time (07:30 UTC on D-1):

  * day-ahead prices of local days <= D-1   (D-1 was cleared in the D-2 auction)
  * wind forecast: NESO/Elexon WINDFOR (transmission) + NESO embedded wind,
    solar forecast: NESO embedded solar, each the latest issue before 07:30 UTC on D-1
  * temperature forecast made 48 h before each valid hour (Open-Meteo previous_day2)
  * national demand forecast (Elexon NDF): latest issue by 09:00 UK time on D-1. The full-day
    issue is published at 08:45 UK time, before both gate closures (EPEX 09:20, N2EX 09:50)
  * demand outturn of local days <= D-2      (D-1 is still running)
  * gas SAP of gas day D-3 (publication time of D-2 unverified -> conservative)
  * UK ETS (UKA) futures settlement of D-2

All gap filling is causal (values from earlier days only).
"""
import os
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
TZ = "Europe/London"


# ---------------------------------------------------------------- raw loaders (UTC)
def load_prices():
    df = pd.read_csv(RAW / "elec_price_hourly_ember.csv")
    t = pd.to_datetime(df["Datetime (UTC)"], format="ISO8601")
    s = pd.Series(df["Price (EUR/MWhe)"].values, index=t, name="price").sort_index()
    live = RAW / "elec_price_hourly_nordpool.csv"
    if live.exists():
        # live extension beyond the (monthly) Ember file: N2EX day-ahead auction results from the
        # Nord Pool data portal, EUR/MWh. On overlapping days both agree to < 4 EUR (mean diff 0.05);
        # Ember stays authoritative wherever it is available.
        lp = pd.read_csv(live, parse_dates=["datetime_utc"]).set_index("datetime_utc")["price"].sort_index()
        s = pd.concat([s, lp[lp.index > s.index.max()]]).rename("price")
    return s


def load_wind_solar():
    w = pd.read_csv(RAW / "wind_forecast_windfor_d1_morning.csv", parse_dates=["target_utc"])
    w = w.set_index("target_utc")["generation"].rename("wind_tx").resample("h").mean()
    e = pd.read_csv(RAW / "embedded_forecast_d1_morning.csv", parse_dates=["target_utc"])
    e = e.set_index("target_utc")[["emb_wind", "emb_solar"]].resample("h").mean()
    return pd.concat([w, e], axis=1)


def load_demand_outturn():
    """National demand outturn; settlement periods are in UK local time."""
    df = pd.read_csv(RAW / "elec_demand_outturn_hh_neso.csv", usecols=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"])
    day = pd.to_datetime(df["SETTLEMENT_DATE"], format="ISO8601")
    start = day.dt.tz_localize(TZ, ambiguous=False, nonexistent="shift_forward").dt.tz_convert("UTC")
    t = (start + pd.to_timedelta((df["SETTLEMENT_PERIOD"] - 1) * 30, unit="min")).dt.tz_convert(None)
    s = pd.Series(df["ND"].values, index=t).sort_index()
    s = s[~s.index.duplicated()]
    s = s.resample("h").mean()
    live = RAW / "elec_demand_outturn_hh_indo.csv"
    if live.exists():
        # live extension beyond the (infrequently updated) NESO file: Elexon INDO initial national
        # demand outturn, near-identical to NESO ND (mean diff ~7 MW on ~24 GW in the overlap).
        li = pd.read_csv(live, parse_dates=["start_utc"]).set_index("start_utc")["demand"].sort_index()
        li = li[~li.index.duplicated()].resample("h").mean()
        s = pd.concat([s, li[li.index > s.index.max()]])
    return s.rename("demand")


def load_demand_forecast():
    df = pd.read_csv(RAW / "demand_forecast_ndf_d1.csv", parse_dates=["target_utc"])
    return df.set_index("target_utc")["demand"].rename("demand_fc").resample("h").mean()


def load_temperature():
    df = pd.read_csv(RAW / "temperature_forecast_d2_openmeteo.csv")
    return pd.Series(df["temperature"].values, index=pd.to_datetime(df["datetime_utc"], format="ISO8601"),
                     name="temperature")


def load_neighbour_prices():
    """Day-ahead prices of the markets coupled to GB via interconnectors (Ember, EUR/MWh).
    The price of delivery day D-1 clears in the SDAC auction at ~12:45 CET on D-2, so at the
    decision time (07:30 UTC on D-1) all delivery days <= D-1 are known; day D is not."""
    df = pd.read_csv(RAW / "elec_price_hourly_neighbours.csv", parse_dates=["datetime_utc"])
    codes = {"France": "fr", "Netherlands": "nl", "Belgium": "be", "Norway": "no",
             "Ireland": "ie", "Denmark": "dk", "Germany": "de"}
    return {codes[c]: g.set_index("datetime_utc")["price"] for c, g in df.groupby("country")}


def load_eua():
    """EU ETS futures settlement in EUR/t (the UCL file named 'ukets' is in fact EUA)."""
    eua = pd.read_csv(RAW / "carbon_ukets_futures_daily_investing.csv", encoding="utf-8-sig")
    return pd.Series(pd.to_numeric(eua["Price"].astype(str).str.replace(",", "")).values,
                     index=pd.to_datetime(eua["Date"], format="%d/%m/%Y")).sort_index()


def load_ttf():
    """ICE Endex Dutch TTF front-month settlement, EUR/MWh (Yahoo Finance)."""
    df = pd.read_csv(RAW / "gas_ttf_daily_yahoo.csv", parse_dates=["Date"])
    return df.set_index("Date")["ttf"]


def load_wind_outturn():
    """Elexon AGWS actual transmission wind generation (MW, hourly mean). Published ~90 min
    after each half hour; only used with lags >= 2 days (wind forecast-error features)."""
    df = pd.read_csv(RAW / "wind_outturn_hh_elexon.csv", parse_dates=["start_utc"])
    return df.set_index("start_utc")["wind"].resample("h").mean().rename("wind_out")


def load_temperature_multi():
    """Population-weighted temperature forecast (same 48 h previous_day2 rule as load_temperature)
    over four cities: South incl. London 45 %, North England 25 %, Midlands/Wales 20 %, Scotland 10 %."""
    w = {"london": 0.45, "manchester": 0.25, "birmingham": 0.20, "glasgow": 0.10}
    df = pd.read_csv(RAW / "temperature_forecast_d2_multicity.csv")
    t = pd.to_datetime(df["datetime_utc"], format="ISO8601")
    return pd.Series(sum(df[c].values * k for c, k in w.items()), index=t, name="temp_pop")


def load_margin():
    """NESO national surplus forecast for delivery day D (Elexon OCNMFD, MW), issue of D-2 -
    the freshest issue that already contains D (see update_data.update_margin). From mid-2021."""
    m = pd.read_csv(RAW / "margin_forecast_ocnmfd.csv", parse_dates=["delivery_day"])
    return m.set_index("delivery_day")["surplus"]


def build_remit_unavail(days):
    """Unavailable capacity (REMIT outage messages) for the hours of each delivery day D,
    using ONLY messages published before the 07:30 UTC D-1 cutoff; the latest revision per
    message chain (mrid) wins, Dismissed/Inactive chains count as withdrawn. Returns two
    (n, 24) arrays in GW: non-interconnector generation and interconnectors (assetId I_*).
    Forward-looking scarcity: planned outages for D are typically announced days to years
    ahead, so unlike outturn-based features this is complete at the cutoff."""
    df = pd.read_csv(Path.home() / ".cache" / "day-ahead-prediction" / "remit_messages.csv.gz",
                     parse_dates=["publishTime", "eventStartTime", "eventEndTime"])
    df = df.dropna(subset=["eventStartTime", "eventEndTime", "publishTime"])
    df["unavailableCapacity"] = pd.to_numeric(df["unavailableCapacity"], errors="coerce").fillna(0.0)
    # dispatchable generation + interconnectors only: wind/solar unavailability is already
    # embodied in the wind_da forecast, and intermittent assets dominate message volume
    is_ic_all = df["assetId"].astype(str).str.startswith("I_")
    disp = df["fuelType"].isin(["Fossil Gas", "Nuclear", "Biomass", "Hydro Pumped Storage",
                                "Fossil Hard coal", "Fossil Oil", "Hydro Run-of-river and poundage",
                                "Other"])
    df = df[disp | is_ic_all]
    df = df.sort_values("publishTime", kind="stable").reset_index(drop=True)
    pub = df["publishTime"].values
    start = df["eventStartTime"].dt.tz_localize(None).values
    end = df["eventEndTime"].dt.tz_localize(None).values
    unav = df["unavailableCapacity"].values
    active = (df["eventStatus"] == "Active").values
    is_ic = df["assetId"].astype(str).str.startswith("I_").values
    aid = pd.factorize(df["assetId"].astype(str))[0]
    mid = pd.factorize(df["mrid"])[0]
    m = mid.max() + 1
    lat = {k: np.zeros(m, dtype=v) for k, v in
           [("start", "datetime64[ns]"), ("end", "datetime64[ns]")]}
    lat_unav, lat_active, lat_ic, seen = np.zeros(m), np.zeros(m, bool), np.zeros(m, bool), np.zeros(m, bool)
    lat_aid = np.zeros(m, dtype=np.int64)
    n_assets = int(aid.max()) + 1

    out_gen, out_ic = np.zeros((len(days), 24)), np.zeros((len(days), 24))
    j = 0
    hour = np.timedelta64(3600, "s")
    acc = np.zeros((n_assets, 24))                             # per-asset accumulator, reused
    for i, d in enumerate(days):
        cutoff = (pd.Timestamp(d) - pd.Timedelta(days=1) + pd.Timedelta(hours=7, minutes=30)).to_datetime64()
        while j < len(pub) and pub[j] <= cutoff:              # fold in newer revisions
            k = mid[j]
            lat["start"][k], lat["end"][k] = start[j], end[j]
            lat_unav[k], lat_active[k], lat_ic[k], seen[k] = unav[j], active[j], is_ic[j], True
            lat_aid[k] = aid[j]
            j += 1
        # UTC instants of the 24 local hours of day d (handles BST/GMT)
        h0 = pd.date_range(d, periods=24, freq="h", tz="Europe/London").tz_convert("UTC").tz_localize(None).values
        cand = seen & lat_active & (lat_unav > 0) & (lat["start"] < h0[-1] + hour) & (lat["end"] > h0[0])
        if not cand.any():
            continue
        s, e, u, ic, a = lat["start"][cand], lat["end"][cand], lat_unav[cand], lat_ic[cand], lat_aid[cand]
        ov = ((np.minimum(e[:, None], h0[None, :] + hour) - np.maximum(s[:, None], h0[None, :]))
              / hour).clip(0, 1)                               # (chains, 24) overlap fraction
        # overlapping message chains of the SAME asset describe the same steel: take the
        # per-asset maximum per hour instead of summing chains (avoids double counting)
        touched = np.unique(a)
        acc[touched] = 0.0
        np.maximum.at(acc, a, u[:, None] * ov)
        asset_ic = np.zeros(n_assets, bool)
        asset_ic[a] = False
        asset_ic[a[ic]] = True
        out_gen[i] = acc[touched][~asset_ic[touched]].sum(0) / 1e3
        out_ic[i] = acc[touched][asset_ic[touched]].sum(0) / 1e3
        acc[touched] = 0.0
    return out_gen, out_ic


def load_carbon():
    """UK ETS (UKA) futures in GBP/t. Before the first UKA observation (UKA futures started in
    May 2021) the EU ETS series is used, rescaled by the median UKA/EUA ratio of their first 20 common trading days."""
    eua = load_eua()
    path = RAW / "carbon_uka_daily_investing.csv"
    if not path.exists():
        return eua.rename("co2")
    uka = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")["Price"]
    ratio = (uka / eua).dropna().iloc[:20].median()      # first 20 common days only (no later information)
    early = eua[eua.index < uka.index.min()] * ratio
    return pd.concat([early, uka]).rename("co2")


def load_fuel_daily():
    gas = pd.read_csv(RAW / "gas_sap_daily_ons.csv")
    gas = pd.Series(gas["SAP actual day"].values, index=pd.to_datetime(gas["Date"]), name="gas").sort_index()
    co2 = load_carbon()
    days = pd.date_range(min(gas.index.min(), co2.index.min()), max(gas.index.max(), co2.index.max()), freq="D")
    return pd.concat([gas, co2], axis=1).reindex(days).ffill()     # weekends/holidays: last published value


# ---------------------------------------------------------------- UTC hours -> local days x 24
def to_local_days(s):
    """(n_days, 24) frame indexed by UK local date. Spring-forward day: the missing hour is
    interpolated from its neighbours in the same day; autumn day: the repeated hour is averaged."""
    s = s.dropna()
    t = pd.DatetimeIndex(s.index).tz_localize("UTC").tz_convert(TZ)
    g = s.groupby([t.tz_localize(None).normalize(), t.hour]).mean().unstack()
    g = g.reindex(columns=range(24))
    return g.T.interpolate(limit=1, limit_area="inside").T


def causal_fill(df, max_days):
    """Fill missing hours with the same hour of an earlier day (never a later one)."""
    out = df.copy()
    for k in range(1, max_days + 1):
        out = out.fillna(df.shift(k))
    return out


# ---------------------------------------------------------------- feature definitions
HIST_DAYS = 7  # length of the LSTM look-back window (in days)

# per-hour exogenous inputs known for the delivery day D
EXOG = ["wind_da", "solar_da", "temperature", "demand_fc", "demand_d7", "demand_d2", "res_load", "hour_sin", "hour_cos"]
# per-day scalars known at D-1
NEIGHBOURS = ["fr", "nl", "be", "no", "ie", "dk", "de"]          # markets coupled to GB
NEIGH_FEATS = [f"{c}_p_d1" for c in NEIGHBOURS]                  # their D-1 daily mean price
# Experimental extra predictors, off by default (EXTRA_FEATS env var):
#   "1": neighbour prices + rolling GB price level/volatility (7 d). Tested online (runs *_nf,
#        2024-2026): every member got slightly WORSE (hybrid 9.79 -> 9.89) -> off.
#   "2": TTF front-month (D-2), EUA (D-2), wind forecast error of D-2/D-3.
#   "4": population-weighted temperature (4 cities), hourly HDD = max(15.5 - T, 0) and daily
#        HDD lags (heating persistence): temp_pop/hdd hourly, hdd_d1/hdd_mean3 per day.
#   "5": NESO OC2 surplus margin for day D (issue of D-2). Tested: no member improved -> off.
#   "6": REMIT unavailable capacity (planned/unplanned outages incl. interconnectors) for the
#        hours of day D, as published before the cutoff: unavail / unavail_ic in GW.
#   "3": all sets (used by test_leakage.py to audit the superset).
EXTRA_FEATS = os.environ.get("EXTRA_FEATS", "0")
if EXTRA_FEATS == "3":                                           # legacy superset alias
    EXTRA_FEATS = "12456"
EXTRA2_FEATS = ["ttf", "eua", "wind_err_d2", "wind_err_d3"]
TEMP_FEATS_H = ["temp_pop", "hdd"]                               # hourly, for the delivery day
TEMP_FEATS_D = ["hdd_d1", "hdd_mean3"]                           # daily lags (building inertia)
MARGIN_FEATS = ["surplus", "surplus_rel"]                        # "5": NESO OC2 margin, issue of D-2
REMIT_FEATS_H = ["unavail", "unavail_ic"]                        # "6": REMIT outages for day D, GW
DAILY = ["gas", "co2", "fuel_cost", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_holiday", "is_weekend",
         "p_d1_mean", "p_d1_max", "p_d1_min", "p_d1_last"]
if "1" in EXTRA_FEATS:
    DAILY += NEIGH_FEATS + ["p_mean7", "p_std7"]
if "2" in EXTRA_FEATS:
    DAILY += EXTRA2_FEATS
if "4" in EXTRA_FEATS:
    EXOG = EXOG + TEMP_FEATS_H
    DAILY += TEMP_FEATS_D
if "5" in EXTRA_FEATS:
    DAILY += MARGIN_FEATS
if "6" in EXTRA_FEATS:
    EXOG = EXOG + REMIT_FEATS_H
# per-hour price lags
PRICE_LAGS = ["p_d1", "p_d2", "p_d3", "p_d7"]
# hourly history channels fed to the LSTM encoder (days D-7 .. D-1)
HIST = ["price", "wind_da", "solar_da", "temperature", "res_load"]


class DayAheadData:
    """Holds per-day arrays.

    For every local delivery day d (index i):
      y[i]       (24,)                actual prices of day d
      tab[i]     (24, n_tab)          tabular features per hour (for tree models / LEAR)
      hist[i]    (24*HIST_DAYS, n_h)  hourly history D-7 .. D-1 (for the LSTM encoder)
      fut[i]     (24, n_f)            known future inputs of day d (for the LSTM decoder)
    """

    def __init__(self, extend_to=None):
        """extend_to: optional date beyond the last complete price day. Extends the day grid for
        live forecasting: those days get features (which need only information <= D-1) but y = NaN,
        so they are never valid training days. Default behaviour is unchanged."""
        price_all = to_local_days(load_prices())
        price = price_all
        ws = load_wind_solar()
        wind = to_local_days(ws["wind_tx"] + ws["emb_wind"])
        solar = to_local_days(ws["emb_solar"])
        temp = to_local_days(load_temperature())
        demand = to_local_days(load_demand_outturn())
        demand_fc = to_local_days(load_demand_forecast())

        # complete price days define the target range; the first day needs all inputs
        price = price[price.notna().all(1)]
        start = max(price.index.min(), wind.index.min(), solar.index.min(), temp.index.min(), demand.index.min(),
                    demand_fc.index.min())
        end = price.index.max()
        if extend_to is not None:
            end = max(end, pd.Timestamp(extend_to))
        self.days = pd.date_range(start, end, freq="D")
        A = lambda f: f.reindex(self.days).values.astype(np.float64)

        P = A(price)
        # price lags / history input: in live operation (extend_to) the Nord Pool feed misses the
        # last 1-2 hours of day D-1 until the next CET auction page appears, although they cleared
        # a day earlier. Fill such end-of-day gaps (<= 3 h) within the day; y itself stays raw.
        if extend_to is not None:
            P_lag = pd.DataFrame(A(price_all)).ffill(axis=1, limit=3).values.copy()
            P_lag[np.isnan(P_lag).any(1)] = np.nan            # require >= 21 real hours
        else:
            P_lag = P
        wind, solar, temp, demand_fc = (A(causal_fill(f.reindex(self.days), 3)) for f in (wind, solar, temp, demand_fc))
        demand = A(causal_fill(demand.reindex(self.days), 7))
        n = len(self.days)

        # demand proxy for day d: same hour d-7 (same weekday) and d-2 (latest complete day)
        demand_d7 = np.full_like(demand, np.nan); demand_d7[7:] = demand[:-7]
        demand_d2 = np.full_like(demand, np.nan); demand_d2[2:] = demand[:-2]
        # residual load entirely from day-ahead forecasts: demand - wind - solar
        res_load = demand_fc - wind - solar

        lag = lambda k: np.vstack([np.full((k, 24), np.nan), P_lag[:-k]])
        hours = np.arange(24)
        hour_sin = np.tile(np.sin(2 * np.pi * hours / 24), (n, 1))
        hour_cos = np.tile(np.cos(2 * np.pi * hours / 24), (n, 1))

        fuel = load_fuel_daily()
        fuel = fuel.reindex(pd.date_range(fuel.index.min(), max(fuel.index.max(), self.days.max()), freq="D")).ffill()
        gas = fuel["gas"].reindex(self.days - pd.Timedelta(days=3)).values      # gas day D-3
        co2 = fuel["co2"].reindex(self.days - pd.Timedelta(days=2)).values      # settlement of D-2
        uk_hol = holidays.UnitedKingdom(years=range(2020, 2028))
        dow = self.days.dayofweek.values
        doy = self.days.dayofyear.values
        p1 = lag(1)
        daily = {
            "gas": gas, "co2": co2,
            # short-run marginal cost of a CCGT (~50 % efficiency, 0.37 tCO2/MWh), GBP/MWh_el:
            # gas SAP is in p/kWh_th -> x10 = GBP/MWh_th
            "fuel_cost": gas * 10 / 0.50 + 0.37 * co2,
            "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "is_holiday": np.array([d in uk_hol for d in self.days.date], dtype=float),
            "is_weekend": (dow >= 5).astype(float),
            "p_d1_mean": p1.mean(1), "p_d1_max": p1.max(1), "p_d1_min": p1.min(1), "p_d1_last": p1[:, -1],
        }
        if "2" in EXTRA_FEATS:
            # TTF front-month and EUA settlements of trading day D-2 (weekends: last settlement)
            fuel2 = pd.concat([load_ttf(), load_eua().rename("eua")], axis=1)
            fuel2 = fuel2.reindex(pd.date_range(fuel2.index.min(), self.days.max(), freq="D")).ffill()
            daily["ttf"] = fuel2["ttf"].reindex(self.days - pd.Timedelta(days=2)).values
            daily["eua"] = fuel2["eua"].reindex(self.days - pd.Timedelta(days=2)).values
            # wind forecast error (transmission): morning D-1 forecast minus outturn, daily mean in GW
            err = (to_local_days(ws["wind_tx"]) - to_local_days(load_wind_outturn())).mean(axis=1)
            e = err.reindex(self.days).ffill().values / 1e3           # ffill = last complete day, causal
            for k in (2, 3):
                v = np.full(n, np.nan); v[k:] = e[:-k]
                daily[f"wind_err_d{k}"] = v
        if "1" in EXTRA_FEATS:
            # neighbour markets: daily mean of their D-1 price (ffill = last published day, causal)
            for c, s in load_neighbour_prices().items():
                m = to_local_days(s).mean(1).reindex(self.days).ffill().values
                daily[f"{c}_p_d1"] = np.r_[np.nan, m[:-1]]
            # rolling GB price level and volatility over the hours of days D-7 .. D-1
            flat = pd.Series(P.ravel())
            idx = np.arange(n) * 24 - 1                           # position of the last hour of day D-1
            for name, r in (("p_mean7", flat.rolling(168, min_periods=120).mean()),
                            ("p_std7", flat.rolling(168, min_periods=120).std())):
                v = np.full(n, np.nan); v[1:] = r.values[idx[1:]]
                daily[name] = v
        hourly = {
            "wind_da": wind, "solar_da": solar, "temperature": temp,
            "demand_fc": demand_fc, "demand_d7": demand_d7, "demand_d2": demand_d2, "res_load": res_load,
            "hour_sin": hour_sin, "hour_cos": hour_cos,
            "p_d1": lag(1), "p_d2": lag(2), "p_d3": lag(3), "p_d7": lag(7),
        }
        if "4" in EXTRA_FEATS:
            # population-weighted temperature (4 cities, 48 h lead) and heating degrees:
            # hourly for the delivery day, plus daily HDD lags (thermal inertia of buildings -
            # several cold days in a row raise demand more than a single one). All values come
            # from the same previous_day2 forecast series, so every lag is known at the cutoff.
            temp_pop = A(causal_fill(to_local_days(load_temperature_multi()).reindex(self.days), 3))
            hdd = np.maximum(15.5 - temp_pop, 0.0)               # UK degree-day base (15.5 C)
            hourly["temp_pop"], hourly["hdd"] = temp_pop, hdd
            hd = hdd.mean(1)                                     # daily mean HDD
            v1 = np.full(n, np.nan); v1[1:] = hd[:-1]
            v3 = np.full(n, np.nan); v3[3:] = (hd[:-3] + hd[1:-2] + hd[2:-1]) / 3
            daily["hdd_d1"], daily["hdd_mean3"] = v1, v3
        if "6" in EXTRA_FEATS:
            # REMIT: unavailable generation / interconnector capacity for the hours of day D,
            # from outage messages published before the 07:30 UTC D-1 cutoff (causal)
            hourly["unavail"], hourly["unavail_ic"] = build_remit_unavail(self.days)
        if "5" in EXTRA_FEATS:
            # scarcity: OC2 surplus for day D (issue of D-2, causal), absolute and relative to the
            # day's demand forecast peak; gaps (<= a few days) filled from earlier issues only
            sur = load_margin().reindex(self.days).ffill(limit=5).values
            daily["surplus"] = sur / 1e3
            daily["surplus_rel"] = sur / np.maximum(np.nanmax(demand_fc, axis=1), 1.0)
        self.tab_cols = EXOG + PRICE_LAGS + DAILY
        tab = [hourly[c] for c in EXOG + PRICE_LAGS] + [np.repeat(daily[c][:, None], 24, 1) for c in DAILY]
        self.tab = np.stack(tab, axis=-1).astype(np.float32)              # (n, 24, n_tab)

        # future inputs: no GB price info (the encoder history carries that);
        # extra feature sets add their non-price-lag scalars for the decoder
        fut_daily = (DAILY[:9] + (NEIGH_FEATS if "1" in EXTRA_FEATS else [])
                     + (EXTRA2_FEATS if "2" in EXTRA_FEATS else [])
                     + (TEMP_FEATS_D if "4" in EXTRA_FEATS else [])
                     + (MARGIN_FEATS if "5" in EXTRA_FEATS else []))
        self.fut_cols = EXOG + fut_daily
        fut = [hourly[c] for c in EXOG] + [np.repeat(daily[c][:, None], 24, 1) for c in fut_daily]
        self.fut = np.stack(fut, axis=-1).astype(np.float32)               # (n, 24, n_f)

        hist_src = np.stack([P_lag, wind, solar, temp, res_load], axis=-1)      # (n, 24, n_h); rows <= D-1 only
        self.hist_cols = HIST
        hist = np.full((n, 24 * HIST_DAYS, hist_src.shape[-1]), np.nan, dtype=np.float32)
        for i in range(HIST_DAYS, n):
            hist[i] = hist_src[i - HIST_DAYS:i].reshape(-1, hist_src.shape[-1])
        self.hist = hist

        self.y = P.astype(np.float32)
        self.feat_ok = (np.isfinite(self.tab).all((1, 2)) & np.isfinite(self.fut).all((1, 2))
                        & np.isfinite(self.hist).all((1, 2)))      # forecastable (features complete)
        self.valid = self.feat_ok & np.isfinite(self.y).all(1)     # usable for training (price known)

    def index_of(self, day):
        return int(self.days.get_loc(pd.Timestamp(day)))


if __name__ == "__main__":
    d = DayAheadData()
    print("days:", d.days[0].date(), "->", d.days[-1].date(), "n =", len(d.days), "valid =", d.valid.sum())
    print("first valid day:", d.days[d.valid.argmax()].date(), " last valid day:", d.days[np.where(d.valid)[0][-1]].date())
    print("tab", d.tab.shape, "hist", d.hist.shape, "fut", d.fut.shape)
    bad = np.where(~d.valid)[0]
    print("invalid days:", [str(x.date()) for x in d.days[bad]][:20], len(bad))
    print(pd.DataFrame(d.tab[d.valid].reshape(-1, d.tab.shape[-1]), columns=d.tab_cols).describe().T.round(1).to_string())
