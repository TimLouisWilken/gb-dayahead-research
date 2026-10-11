#!/usr/bin/env bash
# Run the complete online simulation and evaluation.
# Each member runs in its own process with a fixed thread budget (16 logical cores in total),
# so the neural networks are not starved by the tree models.
set -euo pipefail
cd "$(dirname "$0")/src"
PY=../.venv/bin/python
CHRONOS_PY=~/.cache/day-ahead-prediction/venv-chronos/bin/python   # separate env, see run_chronos.py
LOG=../outputs/logs
mkdir -p "$LOG"

$PY -W ignore run_online.py naive > "$LOG/naive.log" 2>&1
# wave 1: original hybrid members and benchmarks
OMP_NUM_THREADS=4 $PY -W ignore run_online.py lstm        > "$LOG/lstm.log" 2>&1 &
OMP_NUM_THREADS=4 $PY -W ignore run_online.py lstm_static > "$LOG/lstm_static.log" 2>&1 &
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $PY -W ignore run_online.py lear > "$LOG/lear.log" 2>&1 &
( RF_JOBS=3 $PY -W ignore run_online.py rf_long   > "$LOG/rf_long.log" 2>&1
  RF_JOBS=3 $PY -W ignore run_online.py rf_static > "$LOG/rf_static.log" 2>&1 ) &
RF_JOBS=3 $PY -W ignore run_online.py rf_short > "$LOG/rf_short.log" 2>&1 &
wait
RF_JOBS=-1 $PY -W ignore run_online.py lstm_rf > "$LOG/lstm_rf.log" 2>&1   # needs the lstm forecasts
# wave 2: additional members
for m in tcn transformer dnn ddnn; do
  TORCH_THREADS=3 OMP_NUM_THREADS=3 $PY -W ignore run_online.py $m > "$LOG/$m.log" 2>&1 &
done
# gbm runs with the extra predictors (TTF, EUA, wind forecast error): the only member they
# helped in the online A/B test (10.85 -> 10.79; LSTM got worse, RFs/LEAR neutral-negative)
EXTRA_FEATS=2 OMP_NUM_THREADS=2 $PY -W ignore run_online.py gbm > "$LOG/gbm.log" 2>&1 &
# dedicated GBM tail forecasts (pinball loss): extra QRA regressors for spike quantiles
EXTRA_FEATS=2 OMP_NUM_THREADS=2 $PY -W ignore run_online.py gbm_q10 > "$LOG/gbm_q10.log" 2>&1 &
EXTRA_FEATS=2 OMP_NUM_THREADS=2 $PY -W ignore run_online.py gbm_q90 > "$LOG/gbm_q90.log" 2>&1 &
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY -W ignore run_online.py lear4 > "$LOG/lear4.log" 2>&1 &
OMP_NUM_THREADS=1 $PY -W ignore run_online.py merit > "$LOG/merit.log" 2>&1 &
[ -x "$CHRONOS_PY" ] && $CHRONOS_PY -W ignore run_chronos.py > "$LOG/chronos.log" 2>&1 &
wait
$PY -W ignore evaluate.py | tee "$LOG/evaluate.log"
$PY -W ignore probabilistic.py --refit-every 1 | tee "$LOG/probabilistic.log"   # quantile forecasts
