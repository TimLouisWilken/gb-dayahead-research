# Literature review: online hybrid LSTM + Random Forest for GB day-ahead EPF

Method: WebSearch and WebFetch, run on 2026-10-09. I checked each citation against a publisher, arXiv, RePEc, IEEE or DBLP-type record. For Lago et al. (2021) I read the full text. For most other papers I only saw the abstract or the bibliographic record. Items I could not fully verify are marked **[UNVERIFIED: …]**.

### 1. Hybrid LSTM + tree-ensemble architectures

- We found no peer-reviewed paper that combines an LSTM and a random forest for day-ahead EPF in this way (the search was not exhaustive).
- **Residual / error correction (closest match):** Ghimire et al. (2024, *Applied Energy*) use VMD + CNN-LSTM as the first stage, then a Random Forest models the decomposed forecast errors (the "ERCRF" stage). This beat the same model without error correction on half-hourly Queensland prices. It is the most direct precedent for "RF on LSTM residuals".
- **Stacking or regime switching:** Manfre Jaimes et al. (2023, *Forecasting*) run LSTM and XGBoost side by side and pick the final forecast by a price-spike threshold (Alberta, 1–4 days ahead). Xue et al. (2025, arXiv preprint) use XGBoost for an initial forecast and an LSTM on its residuals (NSW market).
- **Caution:** Lago, De Ridder & De Schutter (2018) compared 27 methods and found that "hybrid models do not outperform their simpler counterparts". Lago et al. (2021) found that well-tuned DNNs beat LEAR, but LEAR is still a very strong and cheap baseline. Any hybrid must be benchmarked against LEAR and an equal-weight average, with Diebold-Mariano tests.

### 2. Recalibration and calibration windows in EPF

- **Lago, Marcjasz, De Schutter & Weron (2021)**, the open benchmark (checked in full text):
  - Models should be recalibrated **daily**.
  - The DNN weights are recalibrated daily on a **4-year** window. Hyperparameters and features are optimised **only once**.
  - The LEAR ensemble is the arithmetic mean over windows of **56, 84, 1092 and 1456 days**.
  - The DNN ensemble is the mean of 4 DNNs from separate hyperparameter runs.
  - Even simple averaging improves accuracy.
- **Marcjasz, Serafin & Weron (2018):** error varies strongly with window length and dataset, so choosing one window in advance is risky. Averaging across windows is robust, and they propose a weighting scheme for it.
- **Hubicka, Marcjasz & Weron (2019):** averaging over windows of 28–728 days beats the ex-post best single window. A **few short windows plus a few long windows** do best.
- **Pesaran, Pick & Pranovich (2013):** under continuous structural change, the MSFE-optimal observation weights are close to **exponential down-weighting**. This is the theoretical basis for recency weighting.
- **Concept drift and the 2021–22 crisis:** Nasiadka, Nitka & Weron (2022, arXiv) choose calibration samples with change-point detection (the Narrowest-Over-Threshold method). Their test period covers the crisis. de Marcos Peirotén et al. (2020) argue that the most relevant calibration data are not always the most recent, because regimes recur.
- **Online probabilistic aggregation for EPF:** Berrisch & Ziel (2023, *J. Econometrics*; 2024, *IJF*) develop "CRPS learning", which is based on Bernstein Online Aggregation (BOA). It gives significant CRPS gains over uniform combination for 24-dimensional day-ahead price forecasts. The R package `profoc` implements it. Brusaferri et al. (2025, *Applied Energy*) recalibrate NN ensembles online using conformal prediction.

### 3. Continual and online learning

- **Online forecast combination:**
  - Wintenberger (2017) introduced BOA, exponential weights with a second-order correction that gives optimal rates.
  - Cerqueira et al. (2019) proposed ADE, or "arbitrage". A meta-learner predicts each expert's upcoming loss and weights the experts accordingly. It is implemented in the R package `tsensembler`.
- **Streaming forests:**
  - Saffari et al. (2009): online random forests.
  - Lakshminarayanan, Roy & Teh (2014): Mondrian forests. The online version has the same distribution as the batch version.
  - Gomes et al. (2017): Adaptive Random Forest. Each tree has a drift detector and a background tree that replaces it when drift is detected.
  - Gomes et al. (2018): ARF for regression. Implemented in river / scikit-multiflow.
- **Neural nets:**
  - Pham et al. (2023, ICLR), FSNet: fast per-layer adapters for recent changes plus an associative memory for recurring patterns.
  - Rolnick et al. (2019): a small experience-replay buffer greatly reduces catastrophic forgetting.
  - Du (2025, arXiv:2511.17936): on benign time-based streams, replay performs about the same as plain sequential fine-tuning; it clearly reduces forgetting on heterogeneous multi-task streams. arXiv:2510.00809 (2025) studies catastrophic forgetting of time-series foundation models under continual fine-tuning.
  - Salihoglu, Ahmed & Asadi (2025, arXiv; CAISO): an online-updated LSTM cut MSE by about 23% compared with a static LSTM trained once.

### 4. GB/UK-specific EPF, 2021–2026

- Haben, Caudron & Verma (2021): probabilistic GB day-ahead forecasting. The X-model captures spikes better.
- Mishra et al. (2024): Elexon MID half-hourly prices with gas and demand as external inputs, comparing ARIMA, Prophet, XGBoost, CNN, RNN and LSTM.
- Osone & Kodaira (2025, *IEEE Access*): quantile regression for UK prices using demand and generation forecasts over 1,132 training days. MAE is about £18.27/MWh.
- Earlier GB work and Harkin & Liu (2024, arXiv, Irish I-SEM) both find that **same-day gas price and wind generation are the dominant features**, and that feature importance shifted after 2021.
- I found no published online or continually retrained hybrid model for GB.

### Recommended design: "online hybrid LSTM + RF"

1. **Base members**, all producing 24 hourly outputs per day:
   - (a) LSTM with multi-output, as in Lago's DNN.
   - (b) RF or ExtraTrees on the tabular features: price lags d-1, d-2, d-7, gas SAP, UK ETS, BMRS demand, wind and solar forecasts, calendar.
   - (c) A residual hybrid: RF fitted on the LSTM's **out-of-sample** residuals (Ghimire-style). Residuals must come from rolling-origin forecasts, otherwise information leaks.
   - (d) LEAR as a benchmark, and optionally as a member.
2. **Calibration-window diversity:** fit each tree member on a short window (about 8–12 weeks) and a long window (about 2–3 years). Following Hubicka and Lago, this lets the ensemble react to regime shifts without discarding history.
3. **Cadence:**
   - Tree members: full refit every day, which is cheap.
   - LSTM: **daily warm-start fine-tuning** for a few epochs, at a low learning rate, on the newest data mixed with a **replay buffer** sampled from the whole history (stratified by regime or season).
   - LSTM: full retrain monthly or quarterly.
   - Hyperparameters: frozen, re-tuned at most once or twice a year (Lago practice).
4. **Forgetting:** use exponential recency sample weights for the RF and in the LSTM loss (Pesaran et al.). Treat the half-life as a hyperparameter, starting around 90–180 days. Add drift detection, either ADWIN-style per Gomes ARF or change points per Nasiadka. When drift is detected, temporarily increase the weight on short-window members, or trigger a full LSTM retrain.
5. **Adaptive combination:**
   - Use per-hour online convex weights from **BOA or EWA** over the members, updated daily from realised losses. `profoc` implements BOA and CRPS learning.
   - Alternatives: ADE, or inverse rolling-MAE weights.
   - Always keep an equal-weight average as a benchmark, since it is hard to beat.
6. **Evaluation:** use a rolling-origin test covering 2022 and 2023–25. Report MAE, rMAE and RMSE, with Diebold-Mariano tests against LEAR, a naive forecast and the static models.

### References (Harvard)

- Berrisch, J. and Ziel, F. (2023) 'CRPS learning', *Journal of Econometrics*, 237(2), 105221. doi:10.1016/j.jeconom.2021.11.008
- Berrisch, J. and Ziel, F. (2024) 'Multivariate probabilistic CRPS learning with an application to day-ahead electricity prices', *International Journal of Forecasting*. **[UNVERIFIED: volume, pages and DOI not retrieved]**
- Brusaferri, A., Ballarino, A., Grossi, L. and Laurini, F. (2025) 'On-line conformalized neural networks ensembles for probabilistic forecasting of day-ahead electricity prices', *Applied Energy*. doi:10.1016/j.apenergy.2025.126412 (arXiv:2404.02722)
- Cerqueira, V., Torgo, L., Pinto, F. and Soares, C. (2019) 'Arbitrage of forecasting experts', *Machine Learning*, 108, 913–944. doi:10.1007/s10994-018-05774-y
- de Marcos Peirotén, R.A., Bunn, D.W., Bello, A. and Reneses, J. (2020) 'Short-term electricity price forecasting with recurrent regimes and structural breaks', *Energies*. **[UNVERIFIED: volume and DOI not retrieved]**
- Ghimire, S., Deo, R.C., Casillas-Pérez, D. and Salcedo-Sanz, S. (2024) 'Two-step deep learning framework with error compensation technique for short-term, half-hourly electricity price forecasting', *Applied Energy*, 353, 122059. doi:10.1016/j.apenergy.2023.122059
- Gomes, H.M., Bifet, A., Read, J. et al. (2017) 'Adaptive random forests for evolving data stream classification', *Machine Learning*, 106(9–10), 1469–1495. doi:10.1007/s10994-017-5642-8
- Gomes, H.M., Barddal, J.P., Boiko, L.E. and Bifet, A. (2018) 'Adaptive random forests for data stream regression', *ESANN 2018*. **[UNVERIFIED: no DOI; venue confirmed only via scikit-multiflow docs]**
- Haben, S., Caudron, J. and Verma, J. (2021) 'Probabilistic day-ahead wholesale price forecast: a case study in Great Britain', *Forecasting*, 3(3), 596–632. doi:10.3390/forecast3030038
- Harkin, B. and Liu, X. (2024) 'Forecasting day-ahead electricity prices in the Integrated Single Electricity Market…', arXiv:2408.05628 (preprint).
- Hubicka, K., Marcjasz, G. and Weron, R. (2019) 'A note on averaging day-ahead electricity price forecasts across calibration windows', *IEEE Transactions on Sustainable Energy*, 10(1), 321–323. https://ieeexplore.ieee.org/document/8458131 **[DOI string not retrieved]**
- Lago, J., De Ridder, F. and De Schutter, B. (2018) 'Forecasting spot electricity prices: deep learning approaches and empirical comparison of traditional algorithms', *Applied Energy*, 221, 386–405. doi:10.1016/j.apenergy.2018.02.069
- Lago, J., Marcjasz, G., De Schutter, B. and Weron, R. (2021) 'Forecasting day-ahead electricity prices: a review of state-of-the-art algorithms, best practices and an open-access benchmark', *Applied Energy*, 293, 116983. doi:10.1016/j.apenergy.2021.116983 (arXiv:2008.08004; an erratum corrects the LEAR DE metrics)
- Lakshminarayanan, B., Roy, D.M. and Teh, Y.W. (2014) 'Mondrian forests: efficient online random forests', *NeurIPS 27*, 3140–3148. arXiv:1406.2673
- Manfre Jaimes, D., Zamudio López, M., Zareipour, H. and Quashie, M. (2023) 'A hybrid model for multi-day-ahead electricity price forecasting considering price spikes', *Forecasting*, 5(3), 28. https://www.mdpi.com/2571-9394/5/3/28 **[DOI 10.3390/forecast5030028 inferred, not confirmed]**
- Marcjasz, G., Narajewski, M., Weron, R. and Ziel, F. (2023) 'Distributional neural networks for electricity price forecasting', *Energy Economics*, 125, 106843. doi:10.1016/j.eneco.2023.106843
- Marcjasz, G., Serafin, T. and Weron, R. (2018) 'Selection of calibration windows for day-ahead electricity price forecasting', *Energies*, 11(9), 2364. doi:10.3390/en11092364
- Mishra, B.K., Preniqi, V., Thakker, D. and Feigl, E. (2024) 'Machine learning and deep learning prediction models for time-series: a comparative analytical study for the use case of the UK short-term electricity price prediction', *Discover Internet of Things*, 4, 24. doi:10.1007/s43926-024-00075-4
- Nasiadka, J., Nitka, W. and Weron, R. (2022) 'Calibration window selection based on change-point detection for forecasting electricity prices', arXiv:2204.00872. **[ICCS/LNCS venue UNVERIFIED]**
- Osone, Y. and Kodaira, D. (2025) 'Quantile regression for probabilistic electricity price forecasting in the U.K. electricity market', *IEEE Access*, 13, 10083–10093. doi:10.1109/ACCESS.2025.3528450
- Pesaran, M.H., Pick, A. and Pranovich, M. (2013) 'Optimal forecasts in the presence of structural breaks', *Journal of Econometrics*, 177(2), 134–152. doi:10.1016/j.jeconom.2013.04.002
- Pham, Q., Liu, C., Sahoo, D. and Hoi, S.C.H. (2023) 'Learning fast and slow for online time series forecasting', *ICLR 2023*. arXiv:2202.11672
- Rolnick, D. et al. (2019) 'Experience replay for continual learning', *NeurIPS 2019*. arXiv:1811.11682 (co-authors and pages not checked)
- Saffari, A., Leistner, C., Santner, J., Godec, M. and Bischof, H. (2009) 'On-line random forests', *ICCV Workshops*, 1393–1400. doi:10.1109/ICCVW.2009.5457447
- Salihoglu, S., Ahmed, I. and Asadi, A. (2025) 'Adaptive online learning with LSTM networks for energy price prediction', arXiv:2510.16898 (preprint).
- Wintenberger, O. (2017) 'Optimal learning with Bernstein Online Aggregation', *Machine Learning*, 106(1), 119–141. doi:10.1007/s10994-016-5592-6
- Xue, H. et al. (2025) 'LLM-enhanced feature engineering for multi-factor electricity price predictions', arXiv:2505.11890 (preprint).
- Du, W. (2025) 'Mitigating catastrophic forgetting in streaming generative and predictive learning via stateful replay', arXiv:2511.17936 (preprint).

**Not cited, because the title could not be confirmed:** a 2025 *Energy Informatics* stacking paper (doi:10.1186/s42162-025-00572-y). According to a search snippet, it used RF, LSTM, XGBoost and LightGBM as base models with an RF meta-learner, but the article page was behind a login.

### Added during model development

- Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H. and Choo, J. (2022) 'Reversible instance normalization for accurate time-series forecasting against distribution shift', *ICLR 2022*. https://openreview.net/forum?id=cGDAkQo1C0p
- Uniejewski, B., Weron, R. and Ziel, F. (2018) 'Variance stabilizing transformations for electricity spot price forecasting', *IEEE Transactions on Power Systems*, 33(2), 2219–2229. doi:10.1109/TPWRS.2017.2734563
- Nowotarski, J. and Weron, R. (2015) 'Computing electricity spot price prediction intervals using quantile regression and forecast averaging', *Computational Statistics*, 30(3), 791–803. doi:10.1007/s00180-014-0523-0
- Romano, Y., Patterson, E. and Candès, E. (2019) 'Conformalized quantile regression', *NeurIPS 32*. arXiv:1905.03222
- Gibbs, I. and Candès, E. (2021) 'Adaptive conformal inference under distribution shift', *NeurIPS 34*. arXiv:2106.00170
- Gneiting, T. and Raftery, A.E. (2007) 'Strictly proper scoring rules, prediction, and estimation', *Journal of the American Statistical Association*, 102(477), 359–378. doi:10.1198/016214506000001437
- Chernozhukov, V., Fernández-Val, I. and Galichon, A. (2010) 'Quantile and probability curves without crossing', *Econometrica*, 78(3), 1093–1125. doi:10.3982/ECTA7880
- Kupiec, P.H. (1995) 'Techniques for verifying the accuracy of risk measurement models', *Journal of Derivatives*, 3(2), 73–84. doi:10.3905/jod.1995.407942
