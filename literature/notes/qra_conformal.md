# Literature notes: QRA, Factor-QRA, Kath-Ziel CP, Zaffran AgACI

The PDFs rendered as images; text was extracted with pdftotext. Notes come from the main text (appendices skipped except a few Zaffran details).

**Answers to the four project questions**
- **Original QRA setup and gains:** QR with the 12 individual models' point forecasts as regressors gave the best 50% and 90% interval coverage and beat the best individual model (SNARX). No CRPS or pinball score in the paper, only coverage, Christoffersen tests and PI width.
- **Conformal on top of QRA (two-stage):** none of the four papers does it. Kath & Ziel and Maciejowska et al. use CP and QRA as parallel alternatives; Kath & Ziel (sec. 6.1) only suggest as future work that CP intervals might be "averaged"; Zaffran et al. apply ACI/AgACI to a random forest, not to QRA. **The QRA→ACI pipeline has no direct precedent in these four.**
- **ACI gamma:** Zaffran et al. choose gamma by theory, simulation and aggregation. Small gamma is better under weak dependence; optimal gamma is non-monotone in dependence strength.
- **Kath & Ziel, CP vs QRA:** NCP is "equal or even better" than QRA in Winkler, Christoffersen and pinball.

Caveat: Maciejowska et al. contains no GB/UK data (German EPEX DA/ID3, Polish TGE). The only GB item seen: Bordignon et al. (2013) "Combining day-ahead forecasts for British electricity prices" — cited, not read.

---

## 1. Nowotarski & Weron (2013), original QRA working paper

**Citation as printed:** Jakub Nowotarski, Rafał Weron, "Computing electricity spot price prediction intervals using quantile regression and forecast averaging", HSC Research Report HSC/13/12, Hugo Steinhaus Center, Wrocław University of Technology (manuscript dated December 31, 2013). Published as Computational Statistics 30(3), 791–803 (2015).

**Core contribution.** Two-part study of interval forecasts. (1) Empirical PIs built from combined point forecasts (SIMPLE, LAD averaging) give no significant gains. (2) New method (QR, later QRA): quantile regression with the point forecasts of 12 individual models as regressors and the observed price as dependent variable; does not use the models' own PIs. QR-based PIs beat the best individual model, the smoothed nonparametric SNARX. QR is the natural extension of LAD averaging to arbitrary quantiles (LAD = q=0.5).

**Method.** 12 individual models: AR, ARX, p-AR, p-ARX (spike-preprocessed), TAR, TARX, MRJD, MRJDX, IHMAR, IHMARX, SNAR, SNARX. ARX uses log-prices with lags 24, 48, 168, previous day's minimum, log temperature, Mon/Sat/Sun dummies, FPE criterion. Benchmark PIs: empirical quantiles of in-sample residuals. Expanding window from Aug 22, 2010; first prediction Feb 11, 2011; daily recalibration. Weights recursively estimated up to t−24. 24 separate hourly series; Christoffersen tests per hour. Pinball loss for QR; only 50% and 90% two-sided PIs evaluated. Log prices with mean removed.

**Data/test.** Hourly day-ahead LMP, JCPL zone of PJM (US) + NYC temperature, Aug 22, 2010–Jan 14, 2012. PI validation Sep 23, 2011–Jan 14, 2012 (114 days).

**Results (unconditional coverage, %):**

| Nominal | ARX | SNARX | SIMPLE | LAD | QR |
|---|---|---|---|---|---|
| 50% | 69.74 | 56.51 | 58.63 | 56.36 | 53.55 |
| 90% | 96.13 | 94.23 | 94.44 | 93.64 | 92.07 |

All models over-cover (validation period calmer than calibration). Mean PI width 90%: ARX 21.28, SNARX 20.73, SIMPLE 25.73, LAD 26.20, QR 21.10 (median 19.51). Christoffersen per hour: QR not rejected for conditional coverage at 1% except in 2 peak hours; ARX rejected in unconditional coverage in 21/24 hours (50% PI) and 17/24 (90% PI) at 5%; SIMPLE/LAD PIs too wide in hours 11–16. Ranking: QR > SNARX > SIMPLE, LAD > ARX. No CRPS/Winkler/pinball numbers.

**Spikes:** no explicit handling in QR; LAD/pinball loss robust to spikes.

---

## 2. Maciejowska, Serafin & Uniejewski — Factor-QRA

**Citation as printed:** Katarzyna Maciejowska, Tomasz Serafin, Bartosz Uniejewski, "Probabilistic forecasting with a hybrid Factor-QRA approach: Application to electricity trading", arXiv:2303.08565v2, 19 Nov 2024 (preprint submitted to Electric Power Systems Research).

**Core contribution.** Factor-QRA extends PCA-based forecast averaging to probabilistic forecasting. A panel of 673 point forecasts from one ARX-type model (calibration windows 56–728 days) is compressed by PCA into a few factors entering a quantile regression either directly (FQRA) or after being combined into a point forecast (FQRM). Novelties vs the 2016 version: much richer panel (673 vs 32 forecasts); number of factors selected dynamically by BIC (up to 6); cross-sectional standardisation (z-score across window sizes, per time point) of panel and target (sFQRA, sFQRM), letting forecast dispersion drive interval width. Standardised variants are statistically best; also an economic battery evaluation.

**Method.** ARX per hour (OLS): weekday dummies, price lags d−1, d−2, d−7, previous-day min/max/last, EUA, gas, load, wind, solar (hours 9–17); normal-PIT transform. Probabilistic layer: 182-day window. Benchmarks: HS and CP use the average of 6 window forecasts (56, 84, 112, 714, 721, 728); QRA uses those 6 as regressors; QRM their average. Quantiles q=0.01…0.99; PIs at 25 coverage levels 50–98%. CP = split conformal à la Lei et al./Kath & Ziel. Factor-QRA is computed jointly for all hours with a single model (index t=24(d−1)+h); evaluation per hour.

**Data/test.** EPEX DA hourly 2015–2023, test 29.06.2017–31.12.2023 (2377 days; covers COVID + energy crisis); EPEX ID3 and TGE Poland DA 2015–15.08.2019 (778 days). 728-day initial window, then 182 days.

**Results.** No pinball/CRPS/Winkler; average coverage error (ACE) and Christoffersen per hour. HS, CP, QRA, QRM PIs consistently too narrow on all datasets; sFQRA/sFQRM closest to nominal on German DA and Polish DA. Christoffersen (hours of 24 not rejected at 5%), German DA: no method passes at 50% or 80% (authors blame the 2021–23 volatility surge); at 98% only sFQRA/sFQRM pass (19 hours). ID3 50%: CP 19, HS 9, QRA 14, QRM 17, FQRA 3, FQRM 2, sFQRA 20, sFQRM 19. QRA/QRM slightly beat FQRA/FQRM at 50%/80%, FQRA/FQRM beat them at 98%. Unstandardised factor models ≈ QRA/QRM → the gain comes from standardisation.

**Economic (2.5 MW battery, 90% efficiency; buy bid at 95th pct of forecast low-price hour, offer at 5th pct of high-price hour).** Data-driven strategies cut volume by up to ~10% and raise profit per MWh. Best: sFQRA/sFQRM and QRA/QRM in calm/medium regimes and TGE; FQRA/FQRM in the 2021–22 crisis; HS strong on EPEX 2017–2022, poor on TGE and 2023. Statistical coverage accuracy did not map clearly to profit.

**Comparison with QRA:** sFQRA/sFQRM beat QRA/QRM statistically. CP only a parallel benchmark; **no two-stage QRA then CP.**

---

## 3. Kath & Ziel (2021) — Conformal prediction in day-ahead and intraday power markets

**Citation as printed:** Christopher Kath, Florian Ziel, "Conformal Prediction Interval Estimations with an Application to Day-Ahead and Intraday Power Markets", arXiv:1905.07886v2, 17 Sep 2020 (preprint submitted to International Journal of Forecasting; published IJF 37 (2021) 777–799).

**Core contribution.** Introduces inductive conformal prediction to short-term power prices as a model-agnostic second layer on any point forecast. Uses Normalized CP (NCP): score |y − ŷ| divided by the predicted absolute error from a second error-estimation model; interval ŷ ± λ_α·|ε̂|, heteroscedastic yet symmetric. Compares NCP with QRA, an empirical-error-distribution method (E) and a naive benchmark on three markets; a 144-computation path analysis isolates symmetry, random train/calibration sampling and normalisation.

**Method.** Random 75%/25% train/calibration split (exchangeability-based, not time-ordered). 330-day rolling parameterisation window. Point models Naive, LASSO, KNN, SVM on one ARX-style regression (lags 1, 2, 7; previous-day min/max/last; weekday dummies; 3 PCA components of yesterday's 24 prices; load/wind where available); Yeo-Johnson transform per hour. QRA benchmark: naive excluded from inputs, trained on only the last 8 weeks. Each hour separate. 50% PI (25/75) and 90% PI (5/95).

**Data/test.** Nord Pool DA (test 4.7.2013–31.12.2013, 182 days); German EPEX intraday VWAP (831 test days); GEFCom 2014 US zonal prices (test 18.12.2012–17.12.2013). **No GB data.**

**Results (Table 2):**

| Market / model | 50% cov. | 50% Winkler | 90% cov. | 90% Winkler | 90% pinball |
|---|---|---|---|---|---|
| Nord Pool, SVM_NCP | 50.5 | 4.8 | 90.3 | 9.4 | 0.24 |
| Nord Pool, QRA | 44.9 | 4.7 | 80.9 | 10.2 | 0.26 |
| Nord Pool, Naive_E | 60.2 | 6.4 | 95.6 | 15.0 | 0.38 |
| EPEX ID, LASSO_NCP | 52.5 | 13.7 | 90.0 | 26.5 | 0.66 |
| EPEX ID, QRA | 47.2 | 14.5 | 83.3 | 31.5 | 0.78 |
| GEFCom, LASSO_NCP | 48.6 | 24.3 | 87.7 | 52.0 | 1.30 |
| GEFCom, QRA | 48.5 | 26.3 | 87.4 | 48.6 | 1.21 |

NCP vs E for the same point model: Winkler falls ~10–20% on GEFCom and Nord Pool, mostly at 90%; 50% PI worsens in most cases; for EPEX intraday NCP does not help. Pinball by percentile: NCP poor in middle percentiles, better in the outer ones. Christoffersen: QRA pass rates generally lower than NCP's (Nord Pool 90% UC: 16.7% QRA vs 95.8% SVM_NCP). GEFCom hardest market.

**Conclusion on CP vs QRA (sec. 6.1):** NCP is "equal or even better" than QRA on Winkler, Christoffersen and pinball; "QRA remains behind the Conformal Prediction models" in the 50%/90% PI comparison. Path analysis gives an unclear picture; advise running it per problem.

**Spikes:** Tukey IQR outlier removal with imputation worsened coverage/width/Winkler by ~5–10% → outliers kept. CP vulnerable to outliers/tail clustering with limited data and unsuitable for strong structural breaks (exchangeability). Suggest CP intervals might be aggregated (Carlsson et al. 2014, Vovk 2015); do not combine QRA and CP.

---

## 4. Zaffran, Dieuleveut, Féron, Goude & Josse (2022) — AgACI

**Citation as printed:** Margaux Zaffran, Aymeric Dieuleveut, Olivier Féron, Yannig Goude, Julie Josse, "Adaptive Conformal Predictions for Time Series", arXiv:2202.07282v1 [stat.ML], 15 Feb 2022 (EDF R&D, INRIA, École Polytechnique). Published at ICML 2022.

**Core contribution.** Analyses Adaptive Conformal Inference (ACI; Gibbs & Candès 2021) in the time-series setting without distribution shift. Theory: Theorem 3.1 — in the exchangeable case, ACI degrades efficiency linearly in γ vs standard split CP; in the AR(1) residual case ACI improves efficiency for strongly dependent residuals, and the optimal γ depends non-monotonically on the AR coefficient. Method: **AgACI**, parameter-free — K ACI "experts" with different γ aggregated online with BOA (gradient trick, pinball losses at α/2 and 1−α/2 per bound). Applied to French day-ahead prices.

**Gamma: choices and sensitivity.**
- ACI recursion α_{t+1} = α_t + γ(α − 1{y_t ∉ C_t}). Asymptotic validity for any γ, but the bound |avg miscoverage − α| ≤ 2/(γT) loosens for small γ. Large γ makes α_t leave [0,1] → infinite/uninformative intervals.
- Exchangeable case: expected length minimal at γ=0, grows linearly; Gaussian residuals, α=0.1: γ=0.03 → +1.59% length, γ=0.05 → +3.38%.
- AR(1) (T=10^6): φ=0.6 behaves like exchangeable (optimal γ near 0); for high φ strictly positive γ helps; optimal γ roughly 0–0.08 across φ, non-monotone in φ (longer memory needs slower update).
- Toy AR(1) 0.99: γ=0 gives 72.6% coverage; γ=0.01 and 0.05 give ≈90% and 92%.
- ARMA(1,1) simulations (500 reps, α=0.1): γ=0.01 or 0.05 valid for φ<0.95; γ=0.01 struggles at φ=θ=0.99. AgACI (K=30): coverage above 89.8% for all φ, best efficiency among valid methods.

**Setup.** Online sequential split with sliding window; random forests (1000 trees). Baselines OSSCP, offline SCP, EnbPI. French hourly DA prices 2016–2019 (35,064 obs); predictors: forecast consumption, day of week, prices of D−1 and D−7 (24 each); one RF per hour; test on 2019; sliding 3-year window (1.5y train + 1.5y calibration); α=0.1. Prices reached ~800 EUR/MWh in autumn 2016 vs ~40 average in 2019; offline SCP intervals too wide because the calibration set contains peaks; OSSCP "forgets" peaks.

**Results (Figure 7 only, approximate reads — indicative!).** ACI γ=0.05: coverage ≈0.900, median length ≈21 (smallest); ACI γ=0.01: ≈0.905, ≈22; AgACI: ≈0.918, ≈22.8; EnbPI V2: ≈0.920, ≈23.2; ACI γ=0: ≈0.930, ≈24.5; OSSCP ≈0.932, ≈24.7; offline SCP ≈0.935, ≈26. Text: "ACI gives the smallest intervals with a correct coverage, for γ=0.01 and γ=0.05"; "AgACI is more efficient than γ=0 while maintaining validity. Yet it slightly over-covers, and is slightly less efficient than ACI with a well chosen γ." No pinball/CRPS/Winkler; no comparison with QRA.

**Relevance to γ=0.005:** below the tested values (0.01, 0.05); plausible for near-exchangeable calibration. The 89.6% coverage (0.4 pt below target) is consistent with the finite-sample bound for small γ — inference, not stated in the paper.

---

## Cross-paper observations
1. No two-stage QRA→conformal paper among these four; QRA→ACI is not directly benchmarked anywhere here.
2. QRA benchmarks repeatedly under-cover (Nord Pool 90% PI: 80.9%; EPEX ID 83.3%; GEFCom 87.4%; Maciejowska: too narrow on all datasets) — plausible motivation for a conformal layer on top.
3. Pinball in these papers is a PI-averaged figure; scale not comparable with 3.59 EUR/MWh over 19 quantiles in another market/period.
4. Windows: 182 days (Maciejowska), 330 days with an 8-week QRA window (Kath & Ziel), expanding (original QRA), 3-year sliding with sequential split (Zaffran).
