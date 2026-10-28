# Literature review and positioning of the GB day-ahead system

Compiled 2026-10-12; positioning updated 2026-10-28 for the QRA-8 product. This document complements `docs/literature_review.md` (written 2026-10-09, before the model was built, mostly from abstracts) with a **full-text** review of 25 papers. Full-text PDFs are in `literature/pdf/` (not committed, rebuild with `literature/download.sh`); detailed per-paper notes with exact numbers are in `literature/notes/`.

## 1. Method and limits

- Search: web search for each pillar of the system (reviews and benchmarks, forecast combination and online aggregation, QRA and conformal prediction, deep learning, foundation models, spikes and GB, economic value of forecasts). 26 open-access PDFs were downloaded, 25 of them read through their conclusions (appendices mostly skipped). Per paper, the notes record the citation as printed, the contribution, the data and test periods, and the headline numbers.
- **Not read:** `hubicka2018_calibration_windows_wp.pdf` (downloaded, not opened). Papers known only from abstracts or other papers' citations are marked *(secondary)* below: Gibbs & Candès (2021), Bordignon et al. (2013), Nan (2009), Haben et al. (2021), Osone & Kodaira (2025), Gaillard & Goude (2015), Hirsch & Ziel (2026). Full references for the first five are in `docs/literature_review.md` or the notes.
- Some PDFs are preprints; figures can differ from the journal version.
- Percentages marked "(computed)" in the notes are our arithmetic on printed numbers.
- **Comparability warning.** Almost every published number is for another market and period, with another naive benchmark and another CRPS grid. Comparisons below are qualitative unless stated. There is no open GB benchmark.

## 2. What the literature says, by pillar

### 2.1 Evaluation standards (Lago et al. 2021; Weron 2014; Nowotarski & Weron 2018; Maciejowska et al. 2022)

- Test periods of at least one year (the open benchmark uses two); daily recalibration; hyperparameters chosen ex ante; rMAE instead of MAPE/sMAPE; Diebold–Mariano or Giacomini–White tests (univariate per hour, or multivariate on daily loss norms); computation time reported.
- Probabilistic: reliability first (Kupiec, Christoffersen per hour, or Berkowitz), then sharpness with proper scores only (pinball, Winkler, CRPS); never CWC. "Maximise sharpness subject to reliability."
- Nowotarski & Weron show how short tests mislead: over 12 days the neural nets looked excellent, over 365 days they were among the worst.
- Lago's benchmark (5 markets, 2-year tests, rMAE against weekly persistence p(d−7)): best DNN-ensemble rMAE 0.377 (EPEX-DE) to 0.573 (EPEX-BE); no GB market.

### 2.2 Point forecasting and combination

- **Combination is the robust default.** Weron (2014) quotes Hibon & Evgeniou: the gain is lower risk compared with ex-ante selection, not necessarily beating the best model ex post. Nowotarski & Weron (2016): all combinations beat all individual models (SIMPLE 11.26 vs best individual 11.64 WMAE). Lago: ensembles beat single models in all five markets and should stay small (≈4–10 members).
- **Constrained weights beat unconstrained.** Weron (2014), citing Nowotarski et al. (2013): simple average, constrained least squares (CLS) and IRMSE weights beat ARX and ex-ante selection, while unconstrained OLS/NRLS/BMA weights "perform poorly and should be avoided". CLS was best on Nord Pool (WMAE 4.29 vs 4.47 simple, 4.77–5.36 individual).
- **Online aggregation (BOA).** Wintenberger (2017): second-order exponential weights, optimal fast rate in deviation, not only in expectation. Berrisch & Ziel (2023): the quantile loss is convex but not exp-concave, so EWA has no selection guarantee, while BOA has O(log K/t) to the best expert and O(√(log K/t)) to the best convex combination, each up to log log t. BOA > ML-Poly > EWA empirically in both Berrisch–Ziel papers. El Mahtout & Ziel (2026) apply fully adaptive BOA per hour to point EPF (DE-LU, Spain) and describe this as the first such application: ensembles cut MAE 4–11% against their own best individual models and 14–17% against a DNN benchmark.
- **How big is the gain over a uniform average?** Modest when experts are similar: Berrisch & Ziel (2024) get ≈0.8% CRPS (significant) over the uniform average on German data, but 6.5% over the best single expert. Devaine et al. (2013, load): 7–24% better than the best single expert.
- **Forgetting.** Berrisch & Ziel (2024): forgetting contributes most of the gain (CRPS 1.2855 forget-only vs 1.2918 smooth-only vs 1.2845 both); optimal rates "close to 0". El Mahtout & Ziel (2026) use no forgetting.
- **Smoothing weights.** Berrisch & Ziel smooth across quantiles (2021) and, in 2024, also across hours inside the update with a penalty; the fitted smoothing is milder across hours than across quantiles. No paper smooths already-fitted weights across neighbouring delivery hours at serving time.
- **Models.** Multi-output (24-vector) networks beat per-hour networks in all five markets (Marcjasz et al. 2020; rMAE 0.447–0.611). NBEATSx: up to ≈5% over LEAR/DNN, ≈20% over NBEATS (Olivares et al. 2023). LEAR remains a strong, cheap baseline; the asinh transform matters (Lipiecki et al. 2024: LEAR-QRM CRPS 1.350 vs 1.662 in the earlier paper).
- **Foundation models.** Zero-shot Chronos-T5 is poor on day-ahead prices; Chronos-Bolt and Time-MoE are the only zero-shot families on par with classical models, and a simple MSTL is never significantly beaten except in FR (Hornek et al. 2025; DE-LU 2024 MAE: MSTL 20.69, Chronos-Bolt ≈21.5–21.7, ElasticNet 21.23, naive 29.16). Fine-tuned ChronosX/Moirai lead on CRPS in DE-LU 2024, but a cheap NHITS+QRA with fundamentals or cross-market pretraining matches them (Lettner et al. 2026).
- **Fundamentals.** CoPiLinear (Wang et al. 2024) predicts prices from a rolling piecewise-linear supply curve over load rate: MAE 8.61 vs 11.2 (Shanxi), 7.77 vs 8.80 (ISO-NE). Fundamental drivers (gas, EUA, load, RES) are standard in every strong EPF model.

### 2.3 Probabilistic forecasting

- **QRA** (Nowotarski & Weron 2013/2015): quantile regression on a pool of point forecasts; won the top places at GEFCom2014. Best pinball in Nowotarski & Weron (2018): QRA(3) 2.634 vs 2.774 for the best bootstrapped ARX; never significantly worse than any competitor.
- **QRA under-covers.** Reported coverages: 90% PI at 80.9% (Nord Pool), 83.3% (EPEX intraday), 87.4% (GEFCom) in Kath & Ziel; 7.54% / 91.78% at the 5% / 95% levels in Janczura; QRA/QRM "too narrow" on all datasets in Maciejowska et al.; at most 10 of 24 hours pass Kupiec in Marcjasz et al. 2023.
- **Conformal prediction** (Kath & Ziel 2021): normalised CP is "equal or even better" than QRA in Winkler, Christoffersen and pinball in their three markets; CP is weak in the middle percentiles. Its limits: exchangeability, outliers, structural breaks.
- **Adaptive conformal inference** (Zaffran et al. 2022): for small γ the finite-sample bound on coverage error is 2/(γT); in the exchangeable case length grows linearly with γ (γ=0.05 costs +3.4%); under strong residual dependence γ>0 helps; on French day-ahead prices ACI with γ = 0.01 or 0.05 gave the smallest valid intervals (≈90.0–90.5% coverage, read off a plot). AgACI (BOA over several γ) is parameter-free but slightly over-covers.
- **Alternatives.** Isotonic distributional regression (IDR) wins only in calm periods; the vertical average of QRM, CP and IDR beats each of them (≈5% lower CRPS than QRM in Germany 2021; Lipiecki et al. 2024). Expectile regression averaging reaches near-nominal coverage (5.26% / 94.28%) but only after asinh and with a pinball gain of ≈1% (Janczura 2024). Factor-QRA with cross-sectional standardisation passes Christoffersen in more hours than QRA (Maciejowska et al. 2024).
- **End-to-end distributional networks** (Marcjasz et al. 2023): DDNN-JSU CRPS 1.299 vs DNN-QRA 1.399 (≈7%) in a calm 554-day test, but it degrades badly in 2021–2023 (CRPS 5.395 / 13.375 / 5.265, worse than LEAR-QRM; Lipiecki et al.). Single runs vary up to 10%. Intervals at spikes can be extremely wide.
- **No paper in this set puts conformal inference on top of QRA.** Kath & Ziel only suggest averaging CP intervals as future work.

### 2.4 Spikes and GB

- Weron (2014) and Maryniak & Weron (2014): spike probability is roughly exponential in the demand-to-capacity ratio ρ (β ≈ 20 at 2 days ahead, ≈ 15 at 1 week); below 2% for ρ < 0.85; the link breaks at 2 weeks. Four spike filters give 68–194 spikes in ten years, so "spike" has no standard definition. No classifier metrics are reported.
- GB-specific prior art in full text is thin (Weron 2014 review section on UK). Further GB work known only from abstracts: Bordignon et al. (2013, combination), Haben et al. (2021, probabilistic GB), Osone & Kodaira (2025, quantile regression), Mishra et al. (2024). **No GB market in any open benchmark.**
- Standard handling of spikes in strong models: asinh variance-stabilising transform, absolute/pinball loss, robust estimation. Tukey-style outlier removal made coverage worse (Kath & Ziel).

### 2.5 Economic value of forecasts

- Maciejowska et al. (2026; 192 forecasts, DE-LU 2020–2024, battery arbitrage): MAE/RMSE correlate with profit only at about −0.2 to −0.35; curve-shape association Corr-f +0.86 / +0.90 and min–max price deviation MPD −0.97 / −0.89 are far better predictors; LEAR gave the highest profits although a neural net was most accurate.
- Chitsaz et al. (2018, via Maciejowska et al. 2022): storage scheduling captured 62% of perfect-foresight savings; Uniejewski et al. (2018): 0.20 EUR/MWh lower MAE ≈ 90,000 EUR/yr for a 1 GW baseload trader (2016). Wang et al. (2024): share of theoretical maximum revenue α = 0.45–0.51 with a simple purchase strategy.
- Hirsch & Ziel (2026, *secondary*, arXiv:2604.19580): quantile-based trading strategies do not reward honest probabilistic forecasts.

## 3. Positioning of this project

| Component | Literature | This project | Verdict |
|---|---|---|---|
| Test design | ≥1 year, daily recalibration, ex-ante hyperparameters, DM/GW tests, rMAE (Lago 2021) | 2024-01-01 → 2026-09-27 (≈2.75 years), daily recalibration, strict D−1 information set (07:30 UTC; NESO demand forecast ≤ 09:00 UK time), selection on 2024 with 2025–26 holdout, DM p-values in `outputs/dm_pvalues.csv` | Meets the standard. The information set is tighter than the usual noon-D−1 setup (Maciejowska et al. 2022). |
| Accuracy | Best rMAE in the 5-market benchmark 0.377–0.573 against **p(d−7)** | rMAE 0.391 (sel6) / 0.399 (all13) against the **similar-day** naive (d−1 Tue–Fri, d−7 otherwise); MAE 8.56 | Same order as the best published values, but different market and different naive: **no ranking claim**. Lago notes the ranking of models does not depend on the naive, only the scale. |
| Pool and combination | Small, diverse ensembles; constrained weights; BOA per hour newly applied to EPF (2026) | 13 members (+2 tails for QRA); BOA per hour, forget = 1.0 | In line with the literature; 13 members is above Lago's 4–10 guideline, but the selected-6 variant is the better one (8.565 vs 8.73) — consistent with that guideline. |
| BOA settings | Forgetting is the largest gain in Berrisch–Ziel 2024 (German, 8 similar DDNN experts); none in El Mahtout–Ziel 2026 | Grid search on 2024, holdout 2025–26 (DM p = 0.001) picked **no forgetting** | A divergence from Berrisch–Ziel's finding. Hypothesis (untested): with cumulative (no-forgetting) weights and heterogeneous members, extra forgetting only adds variance. |
| Weight smoothing across adjacent hours (±1, serving only) | Smoothing across hours only inside the update (Berrisch–Ziel 2024, penalised) | Selected on 2024, holdout-confirmed, −0.03 MAE | **No direct precedent found in the papers read**; small effect. Closest relative: Berrisch–Ziel's penalised smoothing over marginals. |
| Chronos-Bolt member | Zero-shot Bolt is the best zero-shot TSFM but not better than MSTL (Hornek 2025) | MAE 14.44 alone (README), kept in the pool | Consistent: weak alone, but kept by the 2024 backward-elimination ablation (selected 6). |
| Fundamental member | CoPiLinear, standard exogenous drivers | Merit-order regression, MAE 12.24 alone | Supported. |
| Probabilistic layer | QRA is the standard but under-covers; CP as an alternative; ERA/IDR/Factor-QRA variants | QRA-8 centred (6 selected point members + 2 pinball-loss GBM tails, shifted onto the selected-6 BOA forecast) + adaptive conformal (γ = 0.005): pinball 3.39, 90% coverage 89.4% (QRA-15 before 2026-10-26: 3.59, 89.6%) | QRA→ACI has no precedent in the papers read. After the conformal step, coverage is closer to nominal than the raw QRA coverages reported in Section 2.3 (which is what the step is for). γ = 0.005 is below the values studied by Zaffran et al. (0.01, 0.05). Absolute scores are not comparable across markets. |
| Distributional net | DDNN-JSU strong in calm periods, unstable in crises | Distributional LSTM (JSU): pinball 3.96, 90% coverage 79.3% as a standalone | Matches the literature: own distributions too narrow, useful as QRA regressors. |
| Spike handling | Reserve margin; asinh; robust losses | Evening-ramp regime analysis; GBM tail members; REMIT outages tested and rejected (planned outages anti-correlate with spikes) | The REMIT result contrasts with the older view that capacity margin is the best spike driver (Weron 2014); our test is on 2024+ GB hours. Spike classification itself was not benchmarked against the literature. |
| Economic evaluation | MAE weakly linked to profit; shape/extreme-hour skill matters (Maciejowska 2026) | BESS LP backtest: all13 93.1% (2025) / 93.7% (2026) of perfect foresight; sel6 92.1% / 93.6%; ranking skill (top/bottom-hour hit rates) explains all13 > sel6 despite worse MAE | Independent confirmation of the same mechanism. Absolute shares are not comparable with Chitsaz (62%) or Wang (0.45–0.51): other strategy, other market. |

### Claims we can make and claims we should not make

- **Defensible:** daily-online ensemble with per-hour BOA on a strict D−1 information set (07:30 UTC, demand forecast ≤ 09:00 UK time), evaluated over 2.75 years of GB data with DM tests; rMAE 0.39 against a similar-day naive; probabilistic layer calibrated to within ≈1 point at 50/80/90%; economic evaluation by LP dispatch showing the MAE/ranking distinction.
- **Not defensible without more work:** "better than published methods" (no common benchmark; GB is absent from epftoolbox); "first" claims beyond what is stated in Section 2 (the search was not exhaustive); that forgetting does not help in general (only for this pool and window).
- **Open comparisons that would strengthen the paper trail:** (1) run LEAR/DNN from `epftoolbox` on the same GB data (our own LEAR reimplementation exists; the open DNN does not); (2) evaluate with Lago's naive p(d−7) as well; (3) report Kupiec/Christoffersen per hour and the multivariate GW test next to coverage and pinball; (4) test the vertical average of QRA, CP and IDR (≈5% CRPS gain in Lipiecki et al.) on top of QRA-8; (5) test cross-sectional standardisation (Factor-QRA) for the QRA pool; (6) read the not-yet-read calibration-window paper (Hubicka et al.).

## 4. Why a colleague may find QRA averaging better than BOA

Question raised in this project: a colleague finds quantile regression averaging beats BOA. The literature separates two things.

1. **QRA at q = 0.5 is an unconstrained LAD combination.** The papers read report that unconstrained weights (OLS, NRLS, BMA) "perform poorly and should be avoided" and that constrained/simple schemes win (Weron 2014; Nowotarski & Weron 2016: SIMPLE 11.26, CLS 11.33, LAD 11.48). This agrees with our own direct measurement: QRA-q50 as point forecast 9.08 vs BOA 8.59 MAE (working paper, negative results), so for point forecasts BOA is preferable in our data.
2. **For quantiles QRA is the state of the art, and BOA does not replace it.** BOA aggregates quantile forecasts of experts that already are distributions (CRPS learning, Berrisch & Ziel); QRA builds a distribution from point forecasts. A colleague who compares *probabilistic* scores of QRA with a BOA of distributional experts may legitimately find QRA better, especially if the experts' own distributions are too narrow (our standalone distributional LSTM covers 79% at nominal 90%).
3. Whether QRA or BOA wins also depends on the pool: QRA corrects bias and scale (coefficients unconstrained), BOA assumes unbiased members. This is our reading, not a result of a cited paper.

## 5. References (full texts read)

Local file names are in `literature/pdf/`; notes in `literature/notes/`. Venue and year as printed on the PDF or checked in the paper itself.

**Reviews and best practice**
- Lago, J., Marcjasz, G., De Schutter, B., Weron, R. (2021) Forecasting day-ahead electricity prices: a review of state-of-the-art algorithms, best practices and an open-access benchmark. *Applied Energy* 293, 116983. arXiv:2008.08004. [`lago2021_epf_review_benchmark.pdf`]
- Weron, R. (2014) Electricity price forecasting: a review of the state-of-the-art with a look into the future. *International Journal of Forecasting* 30(4), 1030–1081. HSC/14/07 working-paper version (pp. 18–52 omitted). [`weron2014_epf_review_wp.pdf`]
- Nowotarski, J., Weron, R. (2018) Recent advances in electricity price forecasting: a review of probabilistic forecasting. *Renewable and Sustainable Energy Reviews* 81, 1548–1568. HSC/16/07. [`nowotarski_weron2018_probabilistic_review_wp.pdf`]
- Maciejowska, K., Uniejewski, B., Weron, R. (2022) Forecasting electricity prices. arXiv:2204.11735. [`weron2022_forecasting_electricity_prices.pdf`]

**Combination and online aggregation**
- Wintenberger, O. (2017) Optimal learning with Bernstein Online Aggregation. *Machine Learning* 106, 119–141. arXiv:1404.1356. [`wintenberger2017_boa.pdf`]
- Berrisch, J., Ziel, F. (2023) CRPS learning. *Journal of Econometrics* 237(2), 105221. arXiv:2102.00968. [`berrisch_ziel2021_crps_learning.pdf`]
- Berrisch, J., Ziel, F. (2024/25) Multivariate probabilistic CRPS learning with an application to day-ahead electricity prices. *International Journal of Forecasting*. arXiv:2303.10019. [`berrisch_ziel2024_multivariate_crps.pdf`]
- El Mahtout, B., Ziel, F. (2026) Electricity price forecasting: bridging linear models, neural networks and online learning. arXiv:2601.02856. [`bridging_linear_nn_online2026.pdf`]
- Devaine, M., Gaillard, P., Goude, Y., Stoltz, G. (2013) Forecasting electricity consumption by aggregating specialized experts. *Machine Learning* 90(2), 231–260. arXiv:1207.1965. [`devaine2013_sleeping_experts_load.pdf`]
- Nowotarski, J., Weron, R. (2016) To combine or not to combine? Recent trends in electricity price forecasting. HSC/16/01. [`nowotarski_weron2016_combine_or_not_wp.pdf`]

**Probabilistic forecasting and conformal prediction**
- Nowotarski, J., Weron, R. (2013/2015) Computing electricity spot price prediction intervals using quantile regression and forecast averaging. HSC/13/12; *Computational Statistics* 30(3), 791–803. [`nowotarski_weron2013_qra_wp.pdf`]
- Maciejowska, K., Serafin, T., Uniejewski, B. (2024) Probabilistic forecasting with a hybrid Factor-QRA approach: application to electricity trading. arXiv:2303.08565. [`maciejowska2024_factor_qra.pdf`]
- Kath, C., Ziel, F. (2021) Conformal prediction interval estimations with an application to day-ahead and intraday power markets. *International Journal of Forecasting* 37, 777–799. arXiv:1905.07886. [`kath_ziel2021_conformal.pdf`]
- Zaffran, M., Dieuleveut, A., Féron, O., Goude, Y., Josse, J. (2022) Adaptive conformal predictions for time series. *ICML 2022*. arXiv:2202.07282. [`zaffran2022_agaci.pdf`]
- Lipiecki, A., Uniejewski, B., Weron, R. (2024) Postprocessing of point predictions for probabilistic forecasting of day-ahead electricity prices: the benefits of using isotonic distributional regression. *Energy Economics* 139, 107934. doi:10.1016/j.eneco.2024.107934. arXiv:2404.02270. [`idr_postprocessing2024.pdf`]
- Janczura, J. (2024) Expectile regression averaging method for probabilistic forecasting of electricity prices. arXiv:2402.07559. [`expectile_regression_averaging2024.pdf`]
- Marcjasz, G., Narajewski, M., Weron, R., Ziel, F. (2023) Distributional neural networks for electricity price forecasting. *Energy Economics* 125, 106843. arXiv:2207.02832. [`marcjasz2023_ddnn.pdf`]

**Deep learning and foundation models**
- Marcjasz, G., Lago, J., Weron, R. (2020) Neural networks in day-ahead electricity price forecasting: single vs. multiple outputs. arXiv:2008.08006. [`marcjasz2020_single_vs_multi_output.pdf`]
- Olivares, K.G., Challu, C., Marcjasz, G., Weron, R., Dubrawski, A. (2023) Neural basis expansion analysis with exogenous variables: forecasting electricity prices with NBEATSx. *International Journal of Forecasting* 39(2), 884–900. arXiv:2104.05522. [`olivares2023_nbeatsx.pdf`]
- Ansari, A.F. et al. (2024) Chronos: learning the language of time series. *TMLR*. arXiv:2403.07815. [`ansari2024_chronos.pdf`]
- Hornek, T., Sartipi, A., Tchappi, I., Fridgen, G. (2025) Benchmarking pre-trained time series models for electricity price forecasting. arXiv:2506.08113. [`pretrained_ts_epf_benchmark2025.pdf`]
- Lettner, J.N., El Ashhab, H., Hagenmeyer, V., Schäfer, B. (2026) Assessing the performance–efficiency trade-off of foundation models in probabilistic electricity price forecasting. arXiv:2604.14739. [`foundation_models_prob_epf2026.pdf`]

**Economic value, simple fundamentals, spikes**
- Maciejowska, K., Lipiecki, A., Uniejewski, B. (2026) Statistical and economic evaluation of forecasts in electricity markets: beyond RMSE and MAE. arXiv:2511.13616. [`beyond_rmse_mae2025.pdf`]
- Wang, L., Liu, J., Zhang, H., Wang, L. (2024/25) Revisiting day-ahead electricity price: simple model save millions. arXiv:2405.14893 (AAAI 2025). [`simple_model_save_millions2024.pdf`]
- Maryniak, P., Weron, R. (2014) Forecasting the occurrence of electricity price spikes in the UK power market. HSC/14/11. [`uk_spike_occurrence2014_wp.pdf`]

**Secondary (not read in full)**
- Gaillard, P., Goude, Y. (2015) Forecasting electricity consumption by aggregating experts; how to design a good set of experts. In: *Modeling and Stochastic Learning for Forecasting in High Dimensions*, Lecture Notes in Statistics 217, Springer. doi:10.1007/978-3-319-18732-7_6
- Hirsch, S., Ziel, F. (2026) Probabilistic forecasting for day-ahead electricity prices, battery trading strategies and the economic evaluation of predictive accuracy. arXiv:2604.19580.

**Downloaded, not read:** Hubicka, K., Marcjasz, G., Weron, R. — averaging forecasts across calibration windows (HSC/18/03). [`hubicka2018_calibration_windows_wp.pdf`]
