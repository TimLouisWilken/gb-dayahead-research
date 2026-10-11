6-seed transformer experiment (2026-10-11)
transformer_s3.npz = second full online run (run_online.py transformer --seed 3),
to be averaged with outputs/preds/transformer.npz for a 6-seed ensemble.

Eval 2024+ single member: 3-seed 9.776 / s3 9.914 / 6-seed mean 9.605 (variance
reduction works solo). In the selected-6 BOA combo: 2024 +0.007 (DM p=0.58),
holdout 2025-26 -0.040 (p=0.03), overall -0.022 (p=0.12).
VERDICT: not adopted - no gain in the 2024 selection year (protocol: select on
2024, holdout only confirms); BOA already averages the seed noise away.
