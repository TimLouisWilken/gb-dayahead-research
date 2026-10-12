# Literature notes: 4 EPF review/benchmark papers

(Note: `weron2014_epf_review_wp.pdf` is the HSC/14/07 working paper in which printed pages 18–52 — the model-by-model review, Sections 3.5–3.9 — are deliberately omitted; everything else was read.)

---

## 1. Lago, Marcjasz, De Schutter & Weron (2021) — `lago2021_epf_review_benchmark.pdf`

**Citation (as printed):** Jesus Lago, Grzegorz Marcjasz, Bart De Schutter, Rafał Weron, "Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark", Applied Energy 293 (2021) 116983. (PDF is the arXiv:2008.08004v2 preprint.)

**Core contribution:** (i) A critical review of state-of-the-art day-ahead EPF algorithms; (ii) a diagnosis of why EPF research is hard to compare (short cherry-picked test periods, no open data, no statistical testing, weak benchmarks); (iii) two open-source reference models — LEAR (lasso-estimated AR with 247 features, asinh variance-stabilizing transform, daily re-estimation; calibration windows of 56, 84, 1092, 1456 days) and a DNN (2 hidden layers, 241 features, hyperparameters/features chosen ex-ante by tree-structured Parzen estimator; 4 runs) plus their arithmetic-mean ensembles; (iv) an open-access benchmark of 5 markets × 6 years with 2-year test periods (`epftoolbox`); (v) a best-practices checklist.

**Datasets/test periods:** 5 markets, each 6 years (2184 days): Nord Pool, PJM COMED, EPEX-BE, EPEX-FR, EPEX-DE. Test = last 104 weeks (2 years): NP & PJM 27.12.2016–24.12.2018; BE & FR 4.1.2015–31.12.2016; DE 4.1.2016–31.12.2017. **No GB dataset.**

**rMAE definition:** MAE divided by the MAE of a naive forecast on the same out-of-sample data. Three candidates: (1) p(d−1,h); (2) p(d−7,h); (3) similar-day (d−1 for Tue–Fri, d−7 for Sat/Sun/Mon). The paper uses naive(2) = weekly persistence p(d−7,h) throughout; model ranking is independent of the naive chosen. "Due to its better properties, rMAE should always be used to evaluate new methods in EPF."

**Headline results — Table 3 (ensembles vs best single models):**

| Market | DNN Ensemble rMAE / MAE | LEAR Ensemble rMAE / MAE | Best single DNN rMAE / MAE | Best single LEAR rMAE / MAE |
|---|---|---|---|---|
| Nord Pool | **0.403** / 1.667 | 0.420 / 1.738 | 0.415 / 1.717 | 0.472 / 1.952 |
| PJM | **0.439** / 2.779 | 0.476 / 3.013 | 0.467 / 2.956 | 0.489 / 3.095 |
| EPEX-BE | **0.573** / 5.821 | 0.604 / 6.140 | 0.597 / 6.066 | 0.649 / 6.594 |
| EPEX-FR | **0.533** / 3.910 | 0.543 / 3.980 | 0.562 / 4.118 | 0.580 / 4.250 |
| EPEX-DE | **0.377** / 3.441 | 0.395 / 3.609 | 0.395 / 3.601 | 0.431 / 3.930 |

sMAPE (DNN-ens / LEAR-ens): NP 4.85/5.01; PJM 11.22/11.98; BE 13.33/14.55; FR 10.98/11.57; DE 14.19/14.74. MAPE is wild for DE (95.76–115.61%, prices near zero) — the paper's own evidence that MAPE is unusable.

GW multivariate tests: the DNN ensemble is statistically significantly better than every other model in all 5 datasets; no individual LEAR is ever significantly better than an individual DNN. Computation per daily recalibration: LEAR 1–10 s, LEAR ensemble 20–25 s, DNN 2–5 min, DNN ensemble 8–20 min.

**Best-practice recommendations:** test period ≥ 1 year (benchmark uses 2); compare against open state-of-the-art models (LEAR/DNN) on open data; use rMAE (MAPE unreliable; sMAPE has undefined mean/infinite variance; MASE breaks with rolling windows/ensembles; linear metrics preferred over RMSE because trading risks are linear in price); mandatory statistical testing — DM and/or GW test, univariate (24 per-hour tests) or multivariate on daily loss-differential norms ||ε_d||_p, p=1 or 2, chessboard heat maps; GW preferred (conditional predictive ability); recalibrate daily; hyperparameters strictly ex-ante; report computation time (>30–60 min per recalibration unsuitable for practice); declare data splits and inputs.

**Forecast combination:** Sec. 5.12 — combining across calibration-window lengths or model diversity "leads to better accuracy"; keep ensembles small (≈4–10 members), especially with heavy-tailed data; new combination schemes must be benchmarked against other ensembles, not only single models. **GB:** not covered. **Spikes:** motivation for asinh VST and ≥1-year tests.

---

## 2. Weron (2014) — `weron2014_epf_review_wp.pdf`

**Citation (as printed):** Rafał Weron, "Electricity price forecasting: A review of the state-of-the-art with a look into the future", International Journal of Forecasting 30(4), 1030–1081 (2014). HSC/14/07 preprint.

**Core contribution:** The field-defining 50-page review of 15 years of EPF. Canonical taxonomy of five model classes — multi-agent, fundamental, reduced-form, statistical, computational intelligence, plus hybrids. Its "look into the future" correctly anticipated: fundamental drivers/seasonality, spike & reserve-margin forecasting, probabilistic forecasts, forecast combination, multivariate factor models, an "EPF-Competition"; postulates objective comparative studies on same datasets / robust error measures / significance testing — the program Lago et al. 2021 implemented.

**Evaluation of point forecasts:** MAPE explodes near zero prices, is small irrespective of error during spikes, negative/uninterpretable for negative prices; recommends scaled errors (seasonal MASE, WMAE, RMSSE). DM test, model confidence set, forecast encompassing. Interval forecasts: Christoffersen (1998) LR tests — UC (≡ Kupiec), independence, conditional coverage; densities: Berkowitz (2001), PIT. Hand-picked single test weeks are "too small to yield conclusive results".

**Forecast combination (Sec. 4.3):** Quotes Hibon & Evgeniou (2005): the advantage of combining "is not that the best possible combinations perform better than the best possible individual forecasts, but that it is less risky in practice to combine than to select". Calls combination the "forgotten art" of EPF; the only empirical EPF evidence then: Nan (2009, PhD, **British** day-ahead prices), Bordignon et al. (2013, **British**), Nowotarski et al. (2013a), Raviv et al. (2013, Nord Pool). Nowotarski et al. (2013a): simple, NRLS, CLS, IRMSE clearly outperform ARX benchmark and best-individual ex-ante selection; unconstrained-weight methods (OLS, NRLS, BMA) "perform poorly and should be avoided"; CLS recommended. Own Nord Pool illustration (6 pure-price models; 30-week test 2013), mean WMAE: individuals 4.77–5.36 vs Simple 4.47, **CLS 4.29 (best)**; in the spikiest week only CLS stayed reasonable (10.48 vs 17.25–21.14). QRA lineage: Nowotarski & Weron (2014a,b) — QR on a point-forecast pool beats best individual + AR benchmark on PJM/Nord Pool.

**UK/GB content (unusually rich):** Cartea et al. (2009) on UK data: virtually all spikes occur at 1-week-ahead demand-to-capacity ratio ρ ∈ [0.908, 0.960]; Maryniak & Weron on APX-UK 2003–2012: spike probability < 2% for ρ < 85%, up to 40–60% near ρ→1; NETA→BETTA (April 2005) raised capacity, reduced spikes; Nan (2009), Bordignon et al. (2013) combination on British prices; Maciejowska & Weron (2013/2014) UK factor models; Gonzalez et al. (2012) hybrid fundamental on APX-UK.

**Spikes:** reserve margin / demand-to-capacity ratio is the single most informative spike predictor; mis-specified long-term seasonal components bias spike intensity estimates.

---

## 3. Nowotarski & Weron (2018) — `nowotarski_weron2018_probabilistic_review_wp.pdf`

**Citation (as printed):** Jakub Nowotarski, Rafał Weron, "Recent advances in electricity price forecasting: A review of probabilistic forecasting", Renewable & Sustainable Energy Reviews 81 (2018) 1548–1568 (WP: HSC/16/07).

**Core contribution:** Tutorial review of probabilistic EPF under the Gneiting paradigm "maximizing sharpness subject to reliability". (i) GEFCom2014 price track (287 contestants; 99 quantiles × 24 hours, pinball loss; top two teams used QRA variants; winner quantMixt combined up to 13 point models incl. random forests and gradient boosting via the **ML-Poly online aggregation forecaster**); (ii) four ways to build probabilistic EPFs: historical simulation (-H), distribution-based (-G), bootstrap (-B), QRA; (iii) complete evaluation toolbox: PICP/ACE, Kupiec, Christoffersen, Berkowitz/PIT; pinball, Winkler, CRPS (CRPS = ∫ pinball dq); DM tests on score differentials ("first paper where the DM test is used for evaluating probabilistic EPFs"); condemns CWC as improper.

**Dataset/test:** GEFCom2014 price-track data; out-of-sample = 365 days, 18.12.2012–17.12.2013. Nine probabilistic models (Naive-H, ARX-H/B, mARX-H/B, NN-G/B, QRA(2), QRA(3)).

**Headline numbers:**
- Reliability (50% PI coverage): best mARX-B 50.08%; QRA(3) 47.23%. 90% PI coverage: best mARX-B **87.44%**, QRA(3) 85.14, NN-B worst 79.86 — almost all models undercover.
- Sharpness (aggregate pinball): Naive-H 3.927 (worst), ARX-B 2.774, **QRA(3) 2.634 (best)**, QRA(2) 2.791. Winkler 90%: QRA(3) best 50.657, Naive-H worst 98.599. QRA's sharpness is not from narrow PIs but from small violation penalties.
- DM tests: **QRA(3) best overall — never significantly worse than any competitor.**
- Cautionary result on test length: over the 12 GEFCom competition days the NNs look excellent, over the full 365 days they are among the worst → "emphasizes the need for long out-of-sample test periods".

**Recommendations:** (1) first formally test reliability — Kupiec + Christoffersen per hour (consecutive hours share one information set), or Berkowitz; (2) rank sharpness with proper scoring rules only — pinball, Winkler, CRPS (avoid CWC); (3) per-hour DM tests on scores. Pinball insight: central quantiles dominate the aggregate pinball, so "it is absolutely crucial to get the point forecasts right"; losses are skewed toward upper quantiles because of spike asymmetry.

---

## 4. Maciejowska, Uniejewski & Weron (2022) — `weron2022_forecasting_electricity_prices.pdf`

**Citation (as printed):** Katarzyna Maciejowska, Bartosz Uniejewski, Rafał Weron, "Forecasting Electricity Prices", arXiv:2204.11735v1 [q-fin.ST], 25 Apr 2022 (for the Oxford Research Encyclopedia of Economics and Finance).

**Core contribution:** Compact modern survey organized around three trends: (1) from point to probabilistic/ensemble forecasting (≤15% of EPF articles concern interval/density forecasts); (2) from small regressions to ML at scale — parsimonious "expert" model vs 247-regressor LEAR, lasso = "golden standard" of EPF shrinkage (ridge clearly worse), DNNs, NBEATSx; (3) from statistical to economic evaluation via trading strategies.

**Market mechanics note:** day-ahead gate closure around noon d−1, so forecast horizons are 12–38 h; at forecast time TSO day-ahead load and RES forecasts are available — this project's 07:30-UTC D-1 information set is tighter than the standard.

**Economic value numbers:** Hong (2015): 1% improvement in load-forecast MAPE saves ~$1.5M/yr for a 5-GW utility, "with price forecasts the savings double"; Uniejewski et al. (2018): 0.20 EUR/MWh MAE decrease → ca. 90,000 EUR/yr extra profit for a 1-GW baseload trader; Delarue et al. (2010): CCGT profits can fall 20% below optimum from price-forecast errors; Chitsaz et al. (2018): storage scheduling captured 62% of perfect-foresight savings; Uniejewski & Weron (2021, 2022): Smoothing QRA beats benchmarks on Kupiec and GW-pinball in all four markets studied and raises trading profits in two.

**Best-practice recommendations:** report MAE and RMSE; MAPE/sMAPE "may lead to absurd results in EPF"; **rMAE (Lago et al. 2021) recommended, naive = P(d−7,h)**; DM test (univariate per-load-period or multivariate ||·||_p) or GW test. Probabilistic: PICP/ACE + Kupiec/Christoffersen/Berkowitz; pinball, Winkler, CRPS; energy score for path forecasts.

**Forecast combination:** two axes — across models (Nan 2009; **Bordignon et al. 2013 on British prices**; Nowotarski et al. 2014) and across calibration windows (Marcjasz et al. 2018; Hubicka et al. 2019); QRA the flagship probabilistic combiner. **GB:** EPF's origins traced to the UK liberalization of the early 1990s; GB-specific: Bordignon, Bunn, Lisi & Nan (2013), Energy Economics 35, 88–103; Maciejowska & Weron (2016), IEEE TPS 31, 994–1005. **Spikes:** spikes "up to two orders of magnitude" + negative prices motivate variance-stabilizing transforms (asinh), long-term seasonal decomposition, robust estimation.

---

## Cross-cutting positioning hooks for the GB/N2EX project
- Project rMAE 0.39 (vs similar-day naive) undercuts every value in Lago's benchmark tables (best anywhere: 0.377, DNN ensemble on EPEX-DE; Nord Pool best 0.403) — but Lago's rMAE denominators are market-specific, so claim "below the best rMAE reported across all five benchmark markets", not a direct ranking.
- The 2.75-year test (2024-01→2026-09) exceeds the ≥1-year minimum and matches the benchmark's 2-year standard.
- BOA per-hour combination is the online-aggregation lineage the literature touches only via ML-Poly inside GEFCom's quantMixt; CLS/simple averaging were the prior state of the art; constrained/regularized weights explicitly favored over unconstrained OLS/BMA.
- QRA on a 13+2-member pool is the canonical recipe ("no limits on the components of X_t"); 89.6% coverage at 90% nominal is better than every model in the Nowotarski-Weron empirical study (best there: mARX-B, 87.44%).
- GB is absent from the open benchmarks; GB-specific prior art is thin: Nan 2009, Bordignon et al. 2013 (combination), Maciejowska & Weron 2013/2016 (factor models/fundamentals), Cartea et al. 2009 + Maryniak & Weron (reserve-margin spike probabilities).
- Evaluation vocabulary to adopt: rMAE + MAE, per-hour and multivariate DM/GW tests, Kupiec/Christoffersen (per hour) or Berkowitz for coverage, aggregate pinball (≈CRPS) and Winkler for sharpness; avoid MAPE and CWC.
