# Transfer to the German day-ahead market (DE-LU): status report

Status as of 2026-10-20. Some runs are still going (transformer in both variants; tcn, ddnn,
lear4 and the GBM quantile members in the TSO variant), so all numbers below are **interim**.

## Verdict so far

The pipeline transfers to Germany without changes to the methods: the same members, the same
hyper-parameters as in GB, the same online protocol and BOA combination. The weak point is the data.
**In Germany no wind and solar forecast is published before the auction closes.** A strict
pre-auction set-up therefore has to build its own renewable forecast from weather data, and that
forecast is clearly worse than the TSO one.

This makes the German variant, for now, **not a strong showcase project**: either it uses data
that is only published after gate closure, or it carries a home-made wind/solar model whose
errors dominate the gap to GB. GB stays the main project. The German variant is kept as a
documented transfer study and is not developed further until a better pre-auction renewable
forecast is available (see the options at the end).

Why GB is different: in GB the system operator publishes its wind and embedded solar forecasts
(NESO WINDFOR and the embedded forecasts) before the decision time, so the GB model never has to
model renewables itself.

## Set-up

| Item | DE choice |
|---|---|
| Market | DE-LU day-ahead (SDAC), 24 hourly prices of local day D; since 2025-10-01 (15-minute products) the hourly mean of the four quarter-hours |
| Decision time | 10:00 Europe/Berlin on D-1; gate closure 12:00 CET |
| Prices, load, generation | SMARD.de (Bundesnetzagentur, CC BY 4.0) |
| Load forecast for D | SMARD/ENTSO-E day-ahead total load forecast; legally due two hours before gate closure and observed on SMARD on the morning of D-1 |
| Wind / solar for D | **strict default:** own proxies from JMA weather forecasts made 48 h ahead (Open-Meteo previous runs), calibrated weekly on realised generation of days ≤ D-2; **comparison only:** TSO day-ahead forecasts (`DE_RENEW=tso`) |
| Temperature | JMA 48 h forecast, population-weighted over six regional centres |
| Fuels | TTF front month (D-2 settlement), EUA (SparkChange CO2.L, before Oct 2021 WisdomTree CARB rescaled) |
| Holidays | German public holidays |
| Code | `MARKET=DE`; `src/update_data_de.py`, `src/data_de.py`, `src/test_leakage_de.py`; outputs in `outputs/de/` (strict) and `outputs/de_tso/` (TSO variant) |

The leakage test (`MARKET=DE python src/test_leakage_de.py`) scrambles every value published
after the decision time, including the realised generation that calibrates the proxies. It passes
on six test days (DST switches, a proxy-refit boundary, the 15-minute switch) and its positive
control detects scrambled D-1 prices.

## When are the TSO wind/solar forecasts published?

The legal deadline is 18:00 on D-1, after the auction. We polled SMARD every ten minutes on
2026-10-20:

- the load forecast for 2026-10-21 was there in the morning;
- the onshore wind, offshore wind and solar forecasts for 2026-10-21 appeared **between 18:05 and
  18:15 CEST**, about six hours after gate closure.

The TSOs themselves have a forecast before the auction (they sell the EEG volumes in it), but it
is not public at that time. The open EPF benchmark for Germany (Lago et al., 2021) uses exactly
these TSO forecasts, so its information set is not available before the auction either.

## Quality of the strict wind/solar proxies

2024 onwards, MAE against realised generation:

| | Own proxy (JMA, 48 h) | TSO forecast | Mean generation |
|---|---|---|---|
| Wind (on- + offshore) | 3,735 MW | 1,388 MW | 15,377 MW |
| Solar | 1,618 MW | 651 MW | 8,828 MW |

The proxy error is about 2.5 times the TSO error. JMA is the only weather model in the archive
that reaches back to 2018 without gaps, and it only provides 10 m wind and cloud cover, with no
radiation and no 100 m wind.

## Interim results (2024-01-01 to 2026-10-20, 1,024 days)

Strict variant, all finished members (transformer still running), EUR/MWh:

| Model | MAE | rMAE |
|---|---|---|
| **Hybrid BOA over the 12 finished members** | **13.49** | **0.461** |
| LEAR 4 windows | 14.89 | 0.509 |
| LEAR | 15.44 | 0.528 |
| Distributional LSTM | 15.56 | 0.532 |
| TCN / GBM / DNN | 15.80 / 15.81 / 15.83 | 0.540 / 0.541 / 0.541 |
| LSTM + RF error correction | 16.00 | 0.547 |
| LSTM online | 16.30 | 0.557 |
| RF long window | 16.68 | 0.571 |
| Merit-order regression | 19.12 | 0.654 |
| RF static (trained once) | 19.98 | 0.683 |
| Chronos-Bolt (zero-shot) | 20.22 | 0.692 |
| LSTM static (trained once) | 21.52 | 0.736 |
| Naive | 29.24 | 1.000 |

The two main GB findings carry over:

- BOA beats LEAR by 13 % (DM p < 0.001) and beats every single model.
- Continual learning pays: RF 16.68 online against 19.98 static, LSTM 16.30 against 21.52.

Strict against TSO variant, on the members finished in both:

| Model | Strict MAE | TSO MAE | Gap |
|---|---|---|---|
| Merit-order | 19.12 | 16.21 | −15 % |
| LEAR | 15.44 | 13.74 | −11 % |
| RF long | 16.68 | 14.56 | −13 % |
| DNN | 15.83 | 13.77 | −13 % |
| LSTM | 16.30 | 14.35 | −12 % |
| LSTM + RF | 16.00 | 13.53 | −15 % |
| GBM | 15.81 | 13.73 | −13 % |
| **BOA over these 8** | **13.58** (rMAE 0.465) | **11.40** (rMAE 0.390) | **−16 %** (DM p < 0.001) |

With the TSO forecasts, the German BOA reaches an rMAE of 0.390, the same level as GB (0.39). So
the method works equally well in Germany, and the whole gap comes from the wind/solar input. That
input, however, is only public after the auction.

## Model fix found during the transfer: LEAR extrapolation guard

In DE, LEAR produced two exploding forecasts (|price| > 10^4 on 2025-05-04 and 2026-05-07). The
cause: in the 182-day winter calibration window the solar input at 20:00 is almost constant, and
in May it suddenly takes a value of hundreds of window standard deviations. The inverse asinh then
turns the linear extrapolation into an extreme price.

The guard clips the standardised inputs to ±10 and the forecast to the window's asinh range ±1.
It is on by default for DE (`LEAR_GUARD`) and off for GB, so the published GB numbers stay
reproducible. Ordinary days are unchanged. The large GB RMSE of `lear4` is probably the same
effect, but GB was not rerun.

## Options to make the German variant worthwhile

1. **Better open weather forecasts:**
   - ICON (DWD) and GFS (NOAA) are archived as 48 h forecasts on Open-Meteo from 2024 on, with
     radiation and 100 m wind;
   - the NOAA GFS archive goes back further;
   - DWD open data (ICON-D2/EU) is available live before 10:00 but kept for only about a day, so
     it would have to be collected from now on.
2. **Better conversion to power:** pvlib and windpowerlib instead of the generic power curve.
3. **Installed capacity from the Marktstammdatenregister** (open data) instead of letting the
   calibration window track the PV build-out.
4. **A commercial wind/solar forecast**, as a real market participant would buy one. It would be
   close to the TSO quality, but it is neither free nor open.

The TSO variant shows the upper bound: perfect access to TSO-quality forecasts is worth about
16 % MAE in the ensemble.

## Licensing

SMARD data are CC BY 4.0. As for GB, raw data (`data/`) and hourly forecast files are not part of
the repository.
