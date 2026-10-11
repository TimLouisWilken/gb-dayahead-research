OC2-margin feature experiment (EXTRA_FEATS=5: OCNMFD surplus of D, issue of D-2, absolute GW
and relative to the demand-forecast peak), 2026-10-11 night.

IMPORTANT data lesson: the daily update revises history (ONS gas re-release, weather re-download),
so the old outputs/preds npz are NOT valid baselines; *_base2 = baselines rerun on the same data
state as the *_m5 runs. (Caught because LEAR, which ignores extra features, differed from its old
npz by up to 32 EUR.)

Burn-in 2023-H2 (MAE): rf_long 13.48 -> 13.14 (PASS -> full online A/B rf_long_basef/rf_long_m5f);
gbm(2+5) 11.63 -> 11.65 (rejected); lstm 11.11 -> 11.44 (rejected); lear 10.90 = 10.90 (sanity ok).
Margin in the spike CLASSIFIER: no AUC gain (0.86/0.93 unchanged - res_load carries the signal).

Full online A/B rf_long (1190 Tage, Eval 2024+, 1006 Tage):
  2024: margin 10.379 vs base 10.375 (+0.004, DM p=0.52)
  2025: margin 10.690 vs base 10.640 (+0.051)
  2026: margin 14.442 vs base 14.642 (-0.200)
  gesamt: 11.632 vs 11.669 (-0.037, DM p=0.31) -> NICHT SIGNIFIKANT
VERDIKT: Margin-Features (EXTRA_FEATS=5) auch fuer rf_long VERWORFEN.
Der Burn-in-Gewinn (13.48->13.14 auf 2023-H2) hat sich nicht generalisiert;
Auswahljahr 2024 zeigt keinerlei Gewinn. Damit ist das OCNMFD-Margin-Set
fuer alle getesteten Member abgelehnt (bleibt aber als Datenquelle fuer
den Spike-Classifier verfuegbar, wo es die AUC ebenfalls nicht verbessert hat).
