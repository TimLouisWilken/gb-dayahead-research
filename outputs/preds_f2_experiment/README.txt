Experiment EXTRA_FEATS=2 (TTF Front-Month D-2, EUA D-2, Windprognosefehler D-2/D-3), Okt 2026.
Online-A/B 2024-01 -> 2026-09 (MAE alt -> neu):
  gbm      10.85 -> 10.79  (besser in jedem Jahr)  -> PRODUKTIV uebernommen
  rf_long  11.58 -> 11.58  neutral
  rf_short 12.65 -> 12.66  neutral
  lstm     10.69 -> 10.99  schlechter (v.a. 2025)
  lstm_rf  10.37 -> 10.54  schlechter
-> Nur das gbm-Mitglied laeuft mit EXTRA_FEATS=2 (siehe run_all.sh).
   gbm_old_feats.npz = der alte gbm-Lauf; preds/gbm.npz ist jetzt der f2-Lauf.
   *_f2.npz hier = die nicht uebernommenen Varianten.
