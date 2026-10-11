Experiment: Zusatz-Prädiktoren (EXTRA_FEATS=1 in src/data.py)
  - Nachbarland-Day-Ahead-Preise (Vortag): FR, NL, BE, NO, IE, DK, DE (Ember)
  - rollierendes GB-Preisniveau/-volatilität über 7 Tage (p_mean7, p_std7)
Online-Rerun von rf_long, rf_short, lstm, lstm_rf (Okt 2026).
Ergebnis (MAE 2024-01 -> 2026-09): alle Member minimal schlechter,
Hybrid EQ 9.79 (alt) -> 9.89 (neu) -> Features standardmaessig deaktiviert.
Reproduzieren: EXTRA_FEATS=1 python src/run_online.py <member> --out <member>_nf
