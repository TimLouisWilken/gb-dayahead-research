#!/usr/bin/env bash
# Re-download the full-text PDFs used for docs/LITERATURE.md.
# PDFs are not committed (arXiv/publisher licenses do not generally permit
# redistribution); run this script to rebuild literature/pdf/ locally.
set -u
cd "$(dirname "$0")" && mkdir -p pdf && cd pdf

dl() { curl -sSL --max-time 120 -A "Mozilla/5.0" -o "$2" "$1" && echo "OK  $2" || echo "FAIL $2"; }

# reviews & best practices
dl https://arxiv.org/pdf/2008.08004 lago2021_epf_review_benchmark.pdf
dl https://alfa.im.pwr.edu.pl/~hugo/RePEc/wuu/wpaper/HSC_14_07.pdf weron2014_epf_review_wp.pdf
dl https://alfa.im.pwr.edu.pl/~hugo/RePEc/wuu/wpaper/HSC_16_07.pdf nowotarski_weron2018_probabilistic_review_wp.pdf
dl https://arxiv.org/pdf/2204.11735 weron2022_forecasting_electricity_prices.pdf
# forecast combination & online learning
dl https://arxiv.org/pdf/1404.1356 wintenberger2017_boa.pdf
dl https://arxiv.org/pdf/2102.00968 berrisch_ziel2021_crps_learning.pdf
dl https://arxiv.org/pdf/2303.10019 berrisch_ziel2024_multivariate_crps.pdf
dl https://arxiv.org/pdf/1207.1965 devaine2013_sleeping_experts_load.pdf
dl https://alfa.im.pwr.edu.pl/~hugo/RePEc/wuu/wpaper/HSC_16_01.pdf nowotarski_weron2016_combine_or_not_wp.pdf
dl https://alfa.im.pwr.edu.pl/~hugo/RePEc/wuu/wpaper/HSC_18_03.pdf hubicka2018_calibration_windows_wp.pdf
dl https://arxiv.org/pdf/2601.02856 bridging_linear_nn_online2026.pdf
# probabilistic forecasting & conformal prediction
dl https://alfa.im.pwr.edu.pl/~hugo/RePEc/wuu/wpaper/HSC_13_12.pdf nowotarski_weron2013_qra_wp.pdf
dl https://arxiv.org/pdf/2303.08565 maciejowska2024_factor_qra.pdf
dl https://arxiv.org/pdf/1905.07886 kath_ziel2021_conformal.pdf
dl https://arxiv.org/pdf/2202.07282 zaffran2022_agaci.pdf
dl https://arxiv.org/pdf/2404.02270 idr_postprocessing2024.pdf
dl https://arxiv.org/pdf/2402.07559 expectile_regression_averaging2024.pdf
dl https://arxiv.org/pdf/2207.02832 marcjasz2023_ddnn.pdf
# deep learning & foundation models
dl https://arxiv.org/pdf/2008.08006 marcjasz2020_single_vs_multi_output.pdf
dl https://arxiv.org/pdf/2104.05522 olivares2023_nbeatsx.pdf
dl https://arxiv.org/pdf/2403.07815 ansari2024_chronos.pdf
dl https://arxiv.org/pdf/2506.08113 pretrained_ts_epf_benchmark2025.pdf
dl https://arxiv.org/pdf/2604.14739 foundation_models_prob_epf2026.pdf
# evaluation, economic value, spikes
dl https://arxiv.org/pdf/2511.13616 beyond_rmse_mae2025.pdf
dl https://arxiv.org/pdf/2405.14893 simple_model_save_millions2024.pdf
dl https://alfa.im.pwr.edu.pl/~hugo/RePEc/wuu/wpaper/HSC_14_11.pdf uk_spike_occurrence2014_wp.pdf
