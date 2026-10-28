# Literature notes: IDR post-processing, Expectile Regression Averaging, Distributional DNN

Percentages marked "(computed)" are arithmetic on printed numbers, not printed in the papers. Figure-only information is flagged.

---

## 1. Lipiecki, Uniejewski, Weron — IDR post-processing (2024)

**Citation as printed:** Arkadiusz Lipiecki, Bartosz Uniejewski, Rafał Weron, "Postprocessing of point predictions for probabilistic forecasting of day-ahead electricity prices: The benefits of using Isotonic Distributional Regression", arXiv:2404.02270v2 [q-fin.ST], 6 Oct 2024 ("Forthcoming in Energy Economics"). Wrocław University of Science and Technology.

**Core contribution.** Compares three ways to turn point forecasts into probabilistic ones: QRA, Conformal Prediction (CP) and Isotonic Distributional Regression (IDR); first use of IDR (and one of the first of CP) in EPF. IDR is the most erratic over time yet contributes most to the ensemble (Shapley). The simple average of the three predictive distributions (LEAR-Ave) is best overall and beats the Marcjasz et al. (2023) DDNN-JSU on two 4.5-year test periods in Germany and Spain.

**Method.** Point forecasts: LEAR (after Lago et al. 2021), windows 56, 84, 1092, 1456 days, one model per hour; regressors: price lags 1, 2, 3, 7; load/RES forecasts at lags 0, 1, 7; EUA, gas, Brent, API2 coal lagged 2 days; weekday dummies; asinh after median/MAD standardisation; LASSO with 7-fold CV repeated 5×. (asinh + repeated CV is why their LEAR-QRM scores 1.350 CRPS vs 1.662 in Marcjasz et al.) Naive: last week's price Mon/Sat/Sun, else yesterday's. Separate distributional models per hour, retrained daily; calibration windows m ∈ {28, 56, 91, 182} days; 99 percentiles; final forecasts = probability ("vertical") average over the 4 windows.
- **QRA:** three variants; single regressor = average of the 4 forecasts (QRM) fastest and most accurate → LEAR-QRM.
- **CP:** symmetric absolute-error conformity score around the averaged LEAR forecast (LEAR-CP).
- **IDR:** non-parametric conditional CDF under a stochastic-order constraint (pool-adjacent-violators), run per LEAR window and averaged (LEAR-IDR).
- **Ensemble:** LEAR-Ave = vertical average of QRM, CP, IDR.
- Compute: LEAR 2:30–3:00 h; QRA 10–15 min; CP 10–15 s; IDR 15–20 s; DDNN-JSU 6:00–6:30 h.
- Scoring: CRPS ≈ average pinball over 99 percentiles; Giacomini-White CPA test; Shapley values; rolling 182-day skill scores vs LEAR-N.

**Data.** Germany and Spain, hourly, 1 Jan 2015–31 Dec 2023 (ENTSO-E; fuels from Investing.com). Out-of-sample 27 Jun 2019–31 Dec 2023 in four subperiods: "2020†" (554 days, same as Marcjasz et al.), 2021, 2022, 2023 (365 days each). Spans COVID and the Ukraine crisis; German prices peak near 800 EUR/MWh and hit the −500 floor on 2 Jul 2023.

**Table 2 — CRPS (avg pinball over 99 percentiles):**

| Germany | 2020† | 2021 | 2022 | 2023 |
|---|---|---|---|---|
| Naive-N | 3.488 | 9.322 | 25.064 | 11.464 |
| LEAR-N | 1.408 | 4.370 | 10.878 | 4.641 |
| LEAR-QRM | 1.350 | 4.189 | 10.651 | 4.422 |
| LEAR-CP | 1.369 | 4.399 | 10.864 | 4.582 |
| LEAR-IDR | 1.422 | 4.389 | 10.926 | 4.336 |
| LEAR-Ave | **1.310** | **3.970** | **10.199** | **4.215** |
| DDNN-JSU | 1.342 | 5.395 | 13.375 | 5.265 |

| Spain | 2020† | 2021 | 2022 | 2023 |
|---|---|---|---|---|
| LEAR-N | 1.018 | 4.160 | 7.412 | 4.735 |
| LEAR-QRM | 0.976 | 4.034 | 7.136 | 4.723 |
| LEAR-CP | 1.014 | 4.208 | 7.371 | 4.699 |
| LEAR-IDR | 0.986 | 4.179 | 7.268 | 4.361 |
| LEAR-Ave | **0.938** | **3.832** | **6.983** | 4.369 |
| DDNN-JSU | 0.989 | 4.627 | 8.299 | 4.379 |

- **IDR vs QRA:** LEAR-QRM beats LEAR-IDR in every subperiod except 2023 (Germany 4.336 vs 4.422; Spain 4.361 vs 4.723). IDR excels in relatively calm periods but is not an all-rounder like QRM. LEAR-Ave vs LEAR-QRM ≈ 5.2% lower in Germany 2021 (computed).
- **Skill scores (Fig. 3, graphical):** IDR most uneven; LEAR-Ave consistently positive, roughly +5 to +10%; DDNN-JSU falls to roughly −40% around the 2022 crisis.
- **CPA test:** LEAR-Ave significantly beats all models on CRPS in both markets; all LEAR-based models significantly beat DDNN-JSU over the full period.
- **Shapley (Fig. 4, read from chart):** Germany IDR ≈0.14, QRM ≈0.08, CP ≈0.065; IDR share >75% in 2023.
- **Reproduction of Marcjasz et al.:** on 2020† LEAR-QRM 1.350 (theirs 1.662), DDNN-JSU 1.342 (theirs 1.304).
- **Table 3 — tails (APS20, aggregate pinball over 20 extreme percentiles):** LEAR-Ave best everywhere except Germany 2020† (DDNN-JSU 0.555 vs 0.575, insignificant) and Spain 2023 (LEAR-IDR 1.856). Germany 2022: QRM 4.579, CP 4.631, IDR 4.985, Ave 4.327, DDNN-JSU 6.321.

**Coverage:** not reported. **Spikes:** only via asinh in the point model; tails evaluated explicitly via APS20; no dedicated spike model.

**Relevance:** QRA is a strong baseline (beats CP and IDR in most subperiods; IDR wins in calm/recent ones); combining QRA+IDR+CP gave ≈5% lower CRPS than QRA alone; the CP here is a symmetric non-adaptive scheme, unlike ACI.

---

## 2. Janczura — Expectile Regression Averaging (ERA) (2024)

**Citation as printed:** Joanna Janczura, "Expectile regression averaging method for probabilistic forecasting of electricity prices", arXiv:2402.07559v1 [stat.AP], 12 Feb 2024. Hugo Steinhaus Center, Wrocław University of Science and Technology.

**Core contribution.** ERA replaces the pinball (L1) loss in QRA with the asymmetric least-squares expectile (L2) loss on a pool of point forecasts; expectiles are converted to quantiles/PIs via Waltrup et al. (2015). On German day-ahead data ERA gives the lowest average pinball score and coverage closer to the 5%/95% nominal levels than quantile methods — after an asinh transform.

**Method.** Pool of 5 ARX models (hourly, one regression per hour): M1 lags 1,2,7 + exogenous; M2 all 7 lagged prices; M3 M2 + previous-day min/max; M4 p-ARX (spike pre-processing, prices beyond mean ± 3σ log-compressed); M5 m-ARX (deviations from weekly mean). Two-year calibration window (first year fits ARX, second year produces point forecasts used to fit weights); moving window; 24 separate per-hour series. Expectile grid τ = 0.001…0.999; conversion by minimising distance to theoretical expectiles. asinh on standardised price; inversion by Monte Carlo. Evaluation: pinball, one-sided DM (5%), coverage at 5% and 95% with Kupiec.

**Data.** German EPEX day-ahead hourly, 1 Jan 2017–31 Dec 2020; validation 2019–2020.

**Table 2, with asinh:**

| | QRA | ERA | Q-hist M1 | EX-hist M1 |
|---|---|---|---|---|
| P5% (%) | 7.54 | **5.26** | 6.64 | 5.16 |
| P95% (%) | 91.78 | 94.28 | 92.40 | 94.12 |
| PS | 2.00 | **1.98** | 2.12 | 2.12 |

Without transform: QRA PS **2.16** beats ERA 2.18 (P5% 8.83 vs 8.02; P95% 93.61 vs 94.16) — the author explains L1 is more robust to outliers than L2, so ERA only wins after variance stabilisation.
- ERA vs QRA with asinh: pinball ≈1% better (computed); coverage improves substantially. DM: ERA beats QRA for 67 percentiles, QRA beats ERA for 8. Historical simulation wins at peak hours 15 and 20.
- Kupiec: hours (of 24) with coverage significantly off: QRA 21 (at 5%) / 22 (at 95%); ERA 0 / 3.

**Spikes:** only via asinh and model M4; ERA's least squares is not robust to outliers. **Caveats:** 2 validation years of German data, ARX point models only.

**Relevance:** plain QRA under-covers at the tails (91.78% at the 95% level) — similar in spirit to why an adaptive conformal layer is added on top.

---

## 3. Marcjasz, Narajewski, Weron, Ziel — Distributional DNN (2023)

**Citation as printed:** Grzegorz Marcjasz, Michał Narajewski, Rafał Weron, Florian Ziel, "Distributional neural networks for electricity price forecasting", arXiv:2207.02832v2 [q-fin.ST], 10 Dec 2022; published Energy Economics 125, 106843 (2023).

**Core contribution.** Distributional deep neural network (DDNN): feed-forward network whose output layer is a parametric distribution — normal (2 parameters) or Johnson's SU (JSU, 4 parameters) — trained by negative log-likelihood; one multivariate network for all 24 hours. First DDNN and first JSU use in EPF. DDNN-JSU beats LEAR-QRA and DNN-QRA on CRPS; heavier-tailed JSU beats normal; aggregating 4 independently tuned networks is crucial for stability.

**Method.** 2 hidden layers; Adam, early stopping (patience 50), up to 1500 epochs; inputs: price lags 1,2,3,7; load and RES forecasts; EUA, coal, gas, oil at T−2; weekday dummies. Hyperparameters via Optuna (2048 iterations, repeated 4 times; hybrid scheme of 13 recalibrations on 28-day batches). Aggregation: pEns (vertical/probability averaging) vs qEns (horizontal/quantile averaging). Benchmarks: naive with bootstrapped residuals; LEAR; point DNNs — converted via QRA or QRM with 182-day rolling calibration, 99 percentiles. CRPS ≈ average pinball over 99 percentiles; Kupiec per hour; DM on CRPS.

**Data.** German day-ahead hourly, 1 Jan 2015–31 Dec 2020; out-of-sample 27 Dec 2018–31 Dec 2020 (first 182 days calibrate QRA → 554-day test set).

**Table 1 (554-day test; Kupiec columns = hours of 24 passed at 5%):**

| Model | MAE | RMSE | CRPS | Kupiec 50% | Kupiec 90% |
|---|---|---|---|---|---|
| naive | 9.336 | 14.358 | 3.585 | 21 | 23 |
| LEAR-QRA | 4.161 | 6.676 | 1.575 | 10 | 8 |
| LEAR-QRM | 4.285 | 6.788 | 1.662 | 6 | 3 |
| DNN-QRA | 3.668 | 5.845 | 1.399 | 6 | 10 |
| DNN-QRM | 3.670 | 5.821 | 1.412 | 9 | 8 |
| DDNN-N-pEns | 3.663 | 5.962 | 1.351 | 2 | 6 |
| DDNN-N-qEns | 3.670 | 5.962 | 1.348 | 13 | 20 |
| DDNN-JSU-pEns | 3.542 | 6.146 | 1.304 | 1 | 4 |
| DDNN-JSU-qEns | 3.564 | 6.174 | **1.299** | 14 | 13 |

- DDNN-JSU-qEns CRPS 1.299 vs DNN-QRA 1.399 (≈7.1% better, computed) and vs LEAR-QRA 1.575 (≈17.5% better, computed); DM: significantly better than every other model. QRA significantly better than QRM for both LEAR and DNN.
- Single DDNN runs span roughly 1.30–1.54 (Fig. 5, graphical); CRPS varies up to 10% run to run.
- **Coverage:** only as counts of hours passing Kupiec. pEns DDNNs too wide (JSU-pEns passes 1 and 4 hours); qEns better (JSU-qEns 14 and 13); QRA models pass at most 10 hours in either column. No raw coverage fractions → no direct comparison to 89.6%.
- **Spikes:** no explicit treatment; JSU skewness/tail weight capture heavy tails. Fig. 4: DDNN-JSU-qEns upper bands reach ≈700 EUR/MWh around a spike where the realised price is ≈190 (very wide intervals at spikes).

**Caveats.** Hyperparameter sets differ a lot between runs and need repeating (costly). Test is 554 days of a relatively calm period; Lipiecki et al. show DDNN-JSU degrades badly in 2021–2023 (CRPS 5.395 / 13.375 / 5.265, all worse than LEAR-QRM).

**Relevance:** QRA on a DNN pool (1.399) is ≈11% better than QRA on LEAR (1.575, computed); DDNN-JSU adds ≈7% more but only in stable periods. Under regime shifts, a post-processing layer on a diverse point-forecast pool (QRA, IDR, CP) held up better than an end-to-end distributional network.

---

## Cross-paper takeaways
- **Scale caveat:** all CRPS numbers are German EUR/MWh averages over 99 percentiles (1.3–1.6 for 2019–2020; 4–11 for 2021–2023) — not directly comparable to the GB 3.39 EUR/MWh average pinball over 19 quantiles for 2024–2026.
- **QRA vs alternatives:** holds up well vs IDR and CP in most subperiods (Lipiecki); under-covers at the tails (Janczura: 7.54% / 91.78% vs 5% / 95%); Marcjasz et al.: passes Kupiec in at most 10 of 24 hours → motivates a conformal correction layer.
- **Combining distributions helps:** Lipiecki's average of QRM, CP, IDR beats the best single method in all subperiods except Spain 2023; averaging runs stabilises DDNN.
- **Variance-stabilising transform matters:** ERA beats QRA only with asinh (Janczura); asinh + better LASSO moved LEAR-QRM from 1.662 to 1.350 (Lipiecki).
