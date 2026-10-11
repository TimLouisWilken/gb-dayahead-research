Temperature-feature experiment (EXTRA_FEATS=4: population-weighted 4-city temperature,
hourly HDD = max(15.5 - T, 0), daily HDD lags d1 / mean d1-d3), 2026-10-11.

Burn-in screen 2023-H2 (MAE, vs the member's production configuration):
  rf_long   13.48 -> 13.25  (passed -> full online A/B, see below)
  rf_short  12.71 -> 12.70  (neutral -> rejected)
  gbm(2+4)  11.63 -> 11.68  (worse  -> rejected)
  lstm      11.08 -> 11.17  (worse  -> rejected)
test_leakage.py with the extended superset (EXTRA_FEATS=3 -> "124"): ALL PASS.

Files here are the burn-in-only runs. rf_long_temp.npz (full 2023-07..2026 run) is added
after the online A/B; decision recorded in the main README.
