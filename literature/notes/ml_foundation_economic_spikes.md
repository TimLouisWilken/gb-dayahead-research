# Literature notes: deep learning, foundation models, economic evaluation, UK spikes (8 papers)

All files in `literature/pdf/`.

---

## 1. Marcjasz, Lago & Weron (2020) — Single vs. multiple outputs
**File:** `marcjasz2020_single_vs_multi_output.pdf`
**Citation (as printed):** Grzegorz Marcjasz, Jesus Lago, Rafał Weron, "Neural networks in day-ahead electricity price forecasting: single vs. multiple outputs", arXiv:2008.08006v1 [stat.AP], 18 Aug 2020; preprint submitted to *Energy Conversion and Management*.

**Core contribution.** Directly compares the two dominant DNN structures for day-ahead EPF: DNN1 (24 independent single-output networks, one per hour, each with its own hyperparameters/features) vs. DNN24 (one network with a vectorized 24-dimensional output mirroring the daily auction). Both are 2-hidden-layer feed-forward DNNs with joint feature + hyperparameter optimization via tree Parzen estimator (241 candidate inputs, 11 feature-selection hyperparameters), 4-year (208-week) rolling calibration, 42-week validation with early stopping; benchmark is LEAR (LASSO-estimated autoregressive, 247 regressors/hour, asinh VST). Finding: the multi-output DNN24 is significantly more accurate AND roughly 20x faster than per-hour DNN1; per-hour specialization promotes overfitting because single-output nets cannot share generalizing representations across hours.

**Data/test periods.** Five open-access epftoolbox datasets (Lago et al. 2021), 2,184 days each; 728-day out-of-sample tests: NP and PJM 27.12.2016–24.12.2018; BE, FR, DE 04.01.2015–31.12.2016.

**Headline numbers (Table 2, rMAE vs. weekly-similar-day naive, all 24 h):**
| dataset | LEAR | DNN24 | DNN1 |
|---|---|---|---|
| NP | 0.482 | **0.472** | 0.472 |
| PJM | 0.492 | **0.477** | 0.495 |
| BE | 0.655 | **0.611** | 0.649 |
| FR | 0.596 | **0.575** | 0.628 |
| DE | 0.452 | **0.447** | 0.516 |

DNN24 best on all five markets; largest DNN1 degradation on DE (0.516 vs 0.447) and FR; DNN1 underperformance concentrated in specific hours (early off-peak), on NP its hour-4 rMAE even exceeded 1 (worse than naive). Giacomini-White CPA test: DNN24 is not outperformed by any model on any dataset; LEAR and DNN1 broadly comparable; all models significantly beat naive. **Relevance:** direct support for multi-output (whole-curve) heads in LSTM/transformer members; LEAR remains a sound benchmark.

---

## 2. Olivares, Challu, Marcjasz, Weron & Dubrawski (2023) — NBEATSx
**File:** `olivares2023_nbeatsx.pdf`
**Citation:** Kin G. Olivares, Cristian Challu, Grzegorz Marcjasz, Rafał Weron, Artur Dubrawski, "Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx", *International Journal of Forecasting* (arXiv:2104.05522v6, 4 Apr 2022).

**Core contribution.** Extends NBEATS (doubly residual stacks of fully connected blocks with backcast/forecast basis expansions) with exogenous variables: an interpretable configuration NBEATSx-I (trend-polynomial, seasonality-Fourier, and exogenous stacks, forecasts decomposable into trend/seasonality/exogenous components) and a generic NBEATSx-G whose exogenous block uses a TCN/WaveNet encoder producing context vectors. H=24, backcast L=168 h. Ensembling (4 models over data-augmentation x early-stopping-set variants, arithmetic mean); MAE loss, ADAM, hyperopt as in Lago et al. (2018a).

**Data/test.** Five hourly markets from epftoolbox, 6 years each, two-year test sets: NP (27-12-2016 to 24-12-2018; load + wind), PJM (same dates), EPEX-FR and EPEX-BE (04-01-2015 to 31-12-2016), EPEX-DE (04-01-2016 to 31-12-2017). Daily recalibration. Metrics MAE, rMAE, sMAPE, RMSE; Giacomini-White test.

**Headline numbers.** NBEATSx improves ~18.77% on average over original NBEATS (and 20.6% over ESRNN) across all metrics/markets; up to ~5% over LEAR and DNN. Vs. DNN ensemble on MAE: +5.38% NP, +2.48% FR, +2.81% DE; non-significant on PJM (0.24%) and BE (1.1%). Ensemble MAE (Table 3): NP — NBEATSx-G **1.58** vs DNN 1.68, LEARx 1.74 (rMAE 0.50 vs 0.53/0.55); PJM — DNN 2.86 vs NBEATSx-G 2.91; BE — DNN 5.87 vs NBEATSx-G 5.95; FR — NBEATSx-I **3.79** vs DNN 3.87 (rMAE 0.64 vs 0.65); DE — NBEATSx-I **3.29** vs DNN 3.41 (rMAE 0.41 vs 0.42). GW tests: no benchmark significantly outperforms NBEATSx-G/-I in any market. Ensembling adds ~3% over single best models. Compute: milliseconds per day-ahead prediction; recalibration ~50% slower than DNN.

---

## 3. Ansari et al. (2024) — Chronos
**File:** `ansari2024_chronos.pdf`
**Citation:** Abdul Fatir Ansari et al., "Chronos: Learning the Language of Time Series", *Transactions on Machine Learning Research* (10/2024), arXiv:2403.07815v3.

**Architecture/method.** Minimalist "language model for time series": mean-scaling (scale s = mean |x| over context) then uniform-bin quantization into a fixed 4,094-bin vocabulary on [-15, +15] (vocab 4096); off-the-shelf T5 encoder-decoder (Mini 20M, Small 46M, Base 200M, Large 710M) trained from scratch with plain cross-entropy (categorical output — "regression via classification"; no time features, no distribution head). Probabilistic by autoregressive sampling (20 paths); context 512, prediction length 64. Trained on 28 datasets (~890K series, ~84B tokens) with TSMixup convex-combination augmentation + 10% KernelSynth synthetic GP data; 200K AdamW steps on 8xA100.

**Zero-shot claims/results.** 42-dataset benchmark: In-domain Benchmark I (15 datasets): Chronos-T5 Large best agg. relative WQL 0.564 and MASE 0.695 (Seasonal Naive = 1.000), significantly beating local statistical models (AutoARIMA 0.876 WQL), task-specific DL (PatchTST 0.601, N-HiTS, DeepAR) and other pretrained models. Zero-shot Benchmark II (27 unseen datasets): Chronos-T5 Large WQL 0.645 / MASE 0.823 — 2nd-4th among all models, within a hair of the best task-specific models *trained on those datasets* (best WQL: TFT 0.639; best MASE: PatchTST 0.810). Fine-tuning Chronos-T5 Small on Benchmark II datasets (~1000 steps) makes it the overall top model (WQL 0.667→0.597). Ablations: performance improves with model size; random init ≥ LLM-weight init; ~10% synthetic data optimal. Limitations noted: struggles with exponential trends, underestimates trend with short context; quantization range limits extreme extrapolation. **Relevance:** lineage of the Chronos-Bolt zero-shot member (Bolt is the faster direct multi-step successor).

---

## 4. Hornek, Sartipi, Tchappi & Fridgen (2025) — Pre-trained TS models for EPF
**File:** `pretrained_ts_epf_benchmark2025.pdf`
**Citation:** Timothée Hornek, Amir Sartipi, Igor Tchappi, Gilbert Fridgen, "Benchmarking Pre-Trained Time Series Models for Electricity Price Forecasting", arXiv:2506.08113v2 [cs.LG], 20 Aug 2025 (© 2025 IEEE), University of Luxembourg SnT.

**Setup.** Zero-shot TSFMs — Chronos-Bolt (Tiny/Mini/Small/Base: 9/21/48/205M), Chronos-T5, Moirai, TimesFM (200M/500M), TimeGPT, Time-MoE — vs. baselines (Naive, SeasonalNaiveDay, SeasonalNaiveWeek), statistical (MSTL, AutoTBATS, AutoMFLES) and ML (ElasticNetCV, KNN, SVR) models. Univariate, price-only; input 1 week; stat/ML retrained daily on 12 weeks. Test: full year 2024 DAA prices, DE-LU, FR, NL, AT, BE (ENTSO-E). Metrics MAE/RMSE/SMAPE + multivariate DM test (daily 1-norm aggregated, p<0.1).

**Headline MAE numbers (EUR/MWh, 2024; Table III):**
| Model | AT | BE | DE | FR | NL |
|---|---|---|---|---|---|
| Naive | 26.35 | 27.15 | 29.16 | 26.89 | 29.31 |
| SeasonalNaiveDay | 22.92 | 23.20 | 27.82 | 20.36 | 26.19 |
| **MSTL** | **17.52** | **17.43** | **20.69** | 16.50 | **19.85** |
| ElasticNet | 18.29 | 18.11 | 21.23 | 17.13 | 20.16 |
| Chronos-Bolt Mini/Small/Base | 17.48/17.56/17.55 | 18.05/18.10/18.27 | 21.61/21.52/21.65 | **16.15/16.02/16.03** | 20.14/20.39/20.35 |
| Chronos-T5 Tiny…Large | 26.12…21.64 | 26.57…23.50 | 31.34…27.72 | 22.70…21.55 | 27.56…25.83 |
| TimesFM 200M/500M | 19.06/26.57 | 19.09/25.00 | 22.36/30.88 | 17.25/21.41 | 21.16/28.28 |
| Time-MoE 50M/200M | 17.96/18.04 | 17.80/18.40 | 22.32/21.61 | 16.69/17.14 | 20.58/20.75 |
| TimeGPT | 20.92 | 20.11 | 23.62 | 18.05 | 22.46 |

**Key findings.** (i) Chronos-Bolt (Mini/Small/Base) and Time-MoE are the only TSFMs on par with traditional models; Chronos-T5 performs poorly. (ii) The humble biseasonal **MSTL is never statistically outperformed by any TSFM** in any country (except FR, where Chronos-Bolt Small significantly beats it). (iii) In DE no TSFM significantly beats MSTL or ElasticNet. (iv) Chronos-Bolt Mini/Small/Base statistically indistinguishable from each other. **Relevance:** justifies the Chronos-Bolt member choice (best zero-shot family for DAA prices) and the finding that it alone does not beat well-configured classical models — hence combination.

---

## 5. Lettner, El Ashhab, Hagenmeyer & Schäfer (2026) — Foundation models in probabilistic EPF
**File:** `foundation_models_prob_epf2026.pdf`
**Citation:** Jan Niklas Lettner, Hadeer El Ashhab, Veit Hagenmeyer, Benjamin Schäfer, "Assessing the Performance–Efficiency Trade-off of Foundation Models in Probabilistic Electricity Price Forecasting", arXiv:2604.14739v1 [cs.LG], 16 Apr 2026; submitted to EDA @ ACM e-Energy '26. KIT.

**Setup.** Day-ahead probabilistic EPF for DE-LU; four models: NHITS+QRA, Transformer-conditioned Normalizing Flow (NF), and two TSFMs: Moirai and ChronosX (Chronos with exogenous-covariate adapters). Data 2018-10-01–2024-12-31, train to end-2022, validation 2023, **test 2024**; cross-border experiments with 14 neighboring bidding zones; zero-/one-/few-shot. Metrics: CRPS, Energy Score, calibration (ECE); DM test with Newey-West HAC on CRPS.

**Headline CRPS results (test 2024, DE-LU).** E1 default configs: fine-tuned TSFMs best — ChronosX large 13.63, ChronosX small 13.86, Moirai base 14.22; zero-shot TSFMs: ChronosX large 15.76, Moirai base 16.63; task-specific from scratch: NHITS+QRA tiny 16.96, NF 24.65-25.34. Feature groups: NHITS+QRA improves monotonically with fundamentals — R5 (load+generation mix) **15.98**; Moirai and NF *degrade* with added covariates. Cross-market: NQ-tiny pretrained on neighboring zones achieves CRPS **13.54 zero-shot** — the task-specific model matches/beats fine-tuned TSFMs; Moirai gains nothing from cross-market data.

**Conclusions.** Fine-tuned TSFMs beat from-scratch task-specific DL on CRPS, but a well-configured, much cheaper NHITS+QRA — especially with informative features or cross-market transfer — matches or surpasses them (TSFM training ~7.7h+ on A100). **Relevance:** supports hybrid pools mixing cheap task-specific models with one TSFM; exogenous handling is a TSFM weak spot (the fundamental/merit-order member covers what Chronos-Bolt can't see).

---

## 6. Maciejowska, Lipiecki & Uniejewski (2026) — Beyond RMSE and MAE
**File:** `beyond_rmse_mae2025.pdf`
**Citation:** Katarzyna Maciejowska, Arkadiusz Lipiecki, Bartosz Uniejewski, "Statistical and economic evaluation of forecasts in electricity markets: beyond RMSE and MAE", arXiv:2511.13616v2 [q-fin.CP], 31 Mar 2026. Wrocław University of Science and Technology.

**Core contribution.** Asks which statistical properties of day-ahead point forecasts matter for economic value, using BESS arbitrage as the application. Pool of 192 hourly forecasts (ARX, NARX shallow NN, LEAR; 8 specifications each; 7 calibration windows 56-1460 days + window-averages). Four evaluation dimensions: accuracy (RMSE, MAE); intraday error dispersion (**Cov-e** = log-det of error covariance matrix); association (**Corr-f** = average daily Spearman correlation between forecast and realized price curves); extreme-hour identification (**MHD** min-max hour deviation; **MPD** min-max price deviation). Two BESS configs: both 3 MWh, BESS-a 3 MW (1-h blocks), BESS-b 1 MW (3-h blocks), efficiencies 98%/97%, operating cost 11.63 EUR/MWh, one full cycle/day.

**Data.** German DE-LU hourly DAA, 2.1.2016–31.12.2024; 1827-day out-of-sample test 2020–2024 (includes COVID + energy crisis).

**Headline correlations (Table 5; Spearman correlation of measure with average profit across the 192-forecast pool):**
- BESS-a: RMSE **-0.20**, MAE **-0.20**; Cov-e **-0.77**; Corr-f **+0.86**; MHD -0.79; **MPD -0.97**.
- BESS-b: RMSE -0.34, MAE -0.35; Cov-e -0.69; Corr-f +0.90; MHD -0.83; MPD -0.89.
- For LEAR forecasts, correlation of RMSE/MAE with profit is nearly zero. MHD collapses (0.85→0.20 within a year) as solar creates two low-price periods (hour-based assessment misleading under duck curves).
- Profit numbers (per MWh traded): Oracle BESS-a 1.42 (2020), 32.03 (2021), 105.84 (2022), 50.25 (2023), 92.57 (2024); choosing best vs. worst forecast changes profits by 16-58% during 2021-2024; NARX most accurate statistically, but **LEAR yields the highest profits**.

**Recommendations.** Don't select forecasts on RMSE/MAE alone; add dispersion (Cov-e) and association (Corr-f) criteria, and MPD for extreme-hour skill; these could also weight ensembles or enter loss functions. **Relevance:** closest methodological support for the project's finding that ranking/top-bottom-hour skill matters more than MAE level for BESS revenue — quantified (MAE↔profit ≈ |0.2|; curve-shape association ≈ 0.86-0.90; MPD ≈ 0.9-0.97).

---

## 7. Wang, Liu, Zhang & Wang (2024) — CoPiLinear ("simple model save millions")
**File:** `simple_model_save_millions2024.pdf`
**Citation:** Linian Wang, Jianghong Liu, Huibin Zhang, Leye Wang, "Revisiting Day-ahead Electricity Price: Simple Model Save Millions", arXiv:2405.14893v2 [cs.LG], 19 Aug 2024 (AAAI 2025). Peking University et al.

**Core contribution.** Argues TSF models fail to exploit the price-formation mechanism. Proposes CoPiLinear: assume short-term stability of the supply curve; express it over **load rate** (supply/capacity); fit an n-segment piecewise-linear supply curve (pwlf) on recent days' equilibrium points; forecast capacity with a Random Forest, take third-party demand forecasts; read the price off the fitted curve. Daily rolling re-fit.

**Data.** Shanxi province, China (96 15-min slots/day; test 2023-04-01–2024-03-31) and ISO New England (hourly; test 2023).

**Headline numbers.** Shanxi: CoPiLinear MAE **8.61 $/MWh** — beats 2nd best (XGBoost 11.2) by **22.9% MAE**; storage-arbitrage revenue α (fraction of theoretical max) 0.512 vs 0.466 best baseline — ~**$7M/yr extra**. ISO-NE: CoPiLinear MAE **7.77** vs Lasso-RF 8.80; α 0.491 vs 0.454 — ~$1M/yr extra. Baselines include SARIMA, VMD-LSTM, DLinear, PatchTST, DNN, Informer, iTransformer etc. ~80.1% of ISO-NE intervals have thermal units marginal, justifying the supply-curve stability assumption. **Relevance:** direct literature support for the merit-order/fundamental member and the economic framing of forecast value; note α ≈ 0.45-0.51 of perfect foresight there (simple quantile strategy) vs. this project's 92-94% (LP dispatch) — different strategy/market, not directly comparable.

---

## 8. Maryniak & Weron (2014) — UK price-spike occurrence
**File:** `uk_spike_occurrence2014_wp.pdf`
**Citation:** Paweł Maryniak, Rafał Weron, "Forecasting the occurrence of electricity price spikes in the UK power market", HSC Research Report HSC/14/11, Hugo Steinhaus Center, Wrocław University of Technology; preprint submitted to Elsevier, August 16, 2014.

**Core contribution.** Studies forward-looking, publicly available UK reserve-margin information for predicting spike occurrence in APX-UK baseload. Extends Cartea et al. (2009): uses the demand-to-capacity ratio ρ = Indicated Demand / Indicated Generation from Elexon's forecasts at horizons 2, 3, 7 and 14 days ahead. Empirical probability study: P(spike | ρ) estimated nonparametrically and fitted with exponential curves.

**Spike definitions (4 identification filters).** (1) CF — Cartea-Figueroa recursive filter on returns (>3σ); (2) VPT — variable price threshold: top 2.5% of deseasonalized prices (S6 wavelet smoothing); (3) RFP — recursive filter on deseasonalized prices (>3σ); (4) RSC — 2-state Markov regime-switching with lognormal spike regime. Sensitivity 19.1.2003-31.12.2012: CF 68 spikes, VPT 91, RSC 170, RFP 194.

**Data.** APX-UK baseload prices 19.1.2003–31.12.2012 (GBP/MWh); analysis window 1.1.2006–31.12.2012 (post-BETTA); Elexon Indicated Demand/Generation.

**Findings/drivers.** Irrespective of spike definition, P(spike) is roughly an **exponentially increasing function of the demand-to-capacity ratio**: P(spike|ρ) ≈ α·exp(β·ρ), β ≈ 20-21 for 2-day-ahead (CF exp(20.91ρ)), β ≈ 15 for 1-week-ahead; curvature increases as horizon shortens. P(spike) < 2% for ρ < 0.85 and well below 1% for ρ < 0.82; rises to ~0.2 (CF) and ~0.5-0.65 (RFP) as ρ → 1. For 2-week horizons the relationship breaks down (market has time to add capacity). No AUC/hit-rate classifier metrics (predates ML-classifier spike papers). **Relevance:** the only GB/UK-specific paper in the set; motivates demand-to-capacity / margin features with an exponential (not threshold) link.

---

### Cross-cutting takeaways
- **Single- vs multi-output:** multi-output DNN24 wins on all 5 markets and generalizes better — aligns with whole-curve heads.
- **NBEATSx vs LEAR/DNN:** ~20% over NBEATS/ESRNN, up to 5% over LEAR/DNN; exogenous inputs decisive.
- **Zero-shot TSFMs on DAA prices:** Chronos-Bolt is the strongest zero-shot family but is never significantly better than MSTL except in FR; fine-tuned TSFMs lead probabilistically, but NHITS+QRA with features matches them at a fraction of the cost.
- **Economic evaluation:** MAE/RMSE correlate with BESS profit only at |ρ|≈0.2-0.35; curve-association Corr-f (+0.86/+0.90) and MPD (-0.97/-0.89) are far better predictors; fundamental supply-curve models can convert accuracy gains into millions $/yr.
- **GB spikes:** spike probability is exponential in the demand-to-capacity ratio with P<1-2% below ρ=0.85.
