# Next experiment: regime-conditional BOA weights

Status: **planned, not implemented** (2026-10-26).

## Motivation

BOA only looks back. Its weights follow the cumulative regret of past days, so after a change in
market conditions it takes days to weeks before the weights have moved. Two kinds of "regime"
have to be kept apart:

| Kind | Examples | Observable on D-1 before the cutoff? | Remedy |
|---|---|---|---|
| Recurring states | windy vs. calm days, high residual load, weekends/holidays, volatile weeks | **yes**, via the D-1 forecasts and price lags | condition the weights on the state (this experiment) |
| Structural breaks | 2021–22 gas crisis, new market design | no | fast adaptation: forgetting / fixed share (already tested: forget = 1.0 won), online-retrained members |

For recurring states the weights do not have to be learned again every time the state changes:
the weights that worked on past days of the *same* state can be used right away. This is the
idea of specialised ("sleeping") experts (Devaine, Gaillard, Goude & Stoltz, 2013, in
`literature/pdf/devaine2013_sleeping_experts_load.pdf`). At EDF it beat the best fixed convex
combination by 5–12 % on load. There is no published result on GB prices yet.

**Hypothesis.** Member skill depends on the state. Examples: `merit` on high residual-load days,
`chronos` and the price-lag models on calm days, `gbm` on windy days. BOA weights conditioned on a
pre-auction state variable beat the unconditional BOA of the selected 6 members (MAE 8.56).

## Variants (pre-registered, at most these four)

All variants use the existing `BOACombiner` (per hour, forget = 1.0, ±1 h serving smoothing)
and only change *which* regret state is updated and served.

| ID | Variant | Description |
|---|---|---|
| R0 | baseline | `hyb_boa_sel`: unconditional BOA, 6 members |
| R1 | bucketed BOA, residual load | one independent `BOACombiner` per tercile of the **daily mean residual-load forecast** of D (`res_load`, `src/data.py:354`) |
| R2 | blended R1 | served weights = (1 − λ)·w_global + λ·w_bucket, λ ∈ {0.25, 0.5, 0.75}. This counters the main risk of R1, that each bucket learns from only ~1/3 of the days |
| R3 | bucketed BOA, wind share | like R1/R2, but the state is the forecast wind + solar share of demand for D |

A state based on recent price volatility (rolling std of D-7…D-1) is a candidate for a later
round. It is left out here to keep the number of variants small.

## Leakage rules

- The state of day D uses only inputs already in the leakage-tested feature set (`res_load` and
  the wind/solar forecasts are issued ≤ 07:30 UTC on D-1).
- **Bucket thresholds are causal:** terciles are computed from days < D only (expanding window from
  the start of the combiner run), never from the full sample.
- The update after day D touches only the bucket that D belonged to, and only after y(D) is known.
- R2's λ and the choice between R1/R2/R3 are made on **2024 only**.

## Protocol

Same as the member-selection ablation (finding 8 in the README):

1. Combine the existing out-of-sample member forecasts (`outputs/preds/*.npz`). No member reruns
   are needed, so a run takes seconds.
2. Pool: `HYBRID_SEL` (main) and `HYBRID_ALL` (robustness check). Bucketing could help the
   13-member pool more, because weak members may be strong in one bucket.
3. **Selection year 2024:** pick the best variant and λ by MAE.
4. **Holdout 2025-01-01 → end:** one-sided DM test, chosen variant vs R0. Also report MAE per year,
   MAE per bucket, MAE in spike hours, and the BESS backtest (`src/bess_backtest.py`).
5. Show the conditional weights per bucket in a figure (analogous to `boa_weights.png`), so it is
   visible whether the members really specialise.

## Decision rule

Adopt for the live system (`src/live.py`) only if all three hold:
- holdout DM p < 0.05 vs R0
- not worse than R0 in any calendar year
- RMSE not worse

Otherwise document it as rejected, like the `EXTRA_FEATS` experiments.

## Implementation sketch

- `src/models.py`: `RegimeBOACombiner(n_experts, n_buckets, lam)`. It holds one `BOACombiner`
  per bucket plus one global one, has `combine(F, s)` / `update(F, y, s)`, and computes causal
  thresholds internally from the states it has seen.
- `src/regime_boa_experiment.py`: loads predictions as in `evaluate.load()`, builds the
  daily state series from `DayAheadData`, runs R0–R3 on both pools, and writes
  `outputs/regime_boa_metrics.csv` and `outputs/figures/regime_boa_weights.png`.
- If adopted: the live state in `outputs/live_state/` needs the per-bucket regret arrays.

## Follow-ups if it works

- Same conditioning for the quantile combination (QRA-15 per bucket, or state × member
  interaction terms in the QRA design matrix).
- Fully learned gating (softmax weights from state features). Held back for now: with ~1,000
  out-of-sample days per hour the risk of overfitting is high, and a gating model cannot
  anticipate a regime that its features do not show (e.g. 2022).
