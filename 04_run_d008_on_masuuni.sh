#!/usr/bin/env bash
# 04_run_d008_on_masuuni.sh
#
# Queue D008 5 folds across both GPUs.
# GPU 0: folds 0, 2, 4 (~3 × 6 h = 18 h)
# GPU 1: folds 1, 3     (~2 × 6 h = 12 h)
# Wall time: 18 h (bottleneck = GPU 0).
#
# Runs in tmux. Each fold's log file is named consistently so the
# existing fold_watcher.sh pattern (see ~/fold_watcher.sh) works as-is.

set -euo pipefail

ML_ENV=${ML_ENV:-$HOME/ml-env}
export nnUNet_raw=${nnUNet_raw:-$HOME/nnunet_raw}
export nnUNet_preprocessed=${nnUNet_preprocessed:-$HOME/nnunet_preprocessed}
export nnUNet_results=${nnUNet_results:-$HOME/nnunet_results}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
source "$ML_ENV/bin/activate"

LOG_DIR=$HOME/nnunet_logs
mkdir -p "$LOG_DIR"

gpu_queue() {
  local gpu=$1
  shift
  local folds=("$@")
  for f in "${folds[@]}"; do
    local log="$LOG_DIR/d8_fold${f}_gpu${gpu}.log"
    echo "[$(date -Is)] gpu=$gpu d=8 fold=$f START" >> "$LOG_DIR/queue_d8_gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES=$gpu nnUNetv2_train 8 3d_fullres "$f" --npz \
      > "$log" 2>&1
    echo "[$(date -Is)] gpu=$gpu d=8 fold=$f DONE exit=$?" >> "$LOG_DIR/queue_d8_gpu${gpu}.log"
  done
}

gpu_queue 0 0 2 4 &
pid0=$!
gpu_queue 1 1 3 &
pid1=$!

echo "[$(date -Is)] D008 queue launched: gpu0 pid=$pid0 (folds 0,2,4), gpu1 pid=$pid1 (folds 1,3)" \
  | tee "$LOG_DIR/queue_d8_master.log"

# Slack ping when starting
if [[ -s "$HOME/.slack_webhook" ]]; then
  URL=$(tr -d '[:space:]' < "$HOME/.slack_webhook")
  curl -s -o /dev/null -X POST -H 'Content-type: application/json' \
    --data '{"text":":arrows_counterclockwise: D008 (sequence-conditioned 3-channel) queue launched on both GPUs, ~18 h wall"}' \
    "$URL" || true
fi

wait "$pid0" "$pid1"

echo "[$(date -Is)] D008 all 5 folds finished" | tee -a "$LOG_DIR/queue_d8_master.log"

if [[ -s "$HOME/.slack_webhook" ]]; then
  URL=$(tr -d '[:space:]' < "$HOME/.slack_webhook")
  curl -s -o /dev/null -X POST -H 'Content-type: application/json' \
    --data '{"text":":white_check_mark: D008 all 5 folds DONE — full PET-MRI HN-cancer queue complete"}' \
    "$URL" || true
fi
