#!/usr/bin/env bash
# 03_run_on_masuuni.sh
#
# Run on Masuuni. Activates ~/ml-env, sets nnU-Net env vars, runs
# plan_and_preprocess, then queues 10 folds (D006 5x + D007 5x) across 2 GPUs.
#
# Usage (on Masuuni):
#   tmux new -s nnunet
#   bash scripts/03_run_on_masuuni.sh preprocess          # one-time
#   bash scripts/03_run_on_masuuni.sh train_all           # main run
#
# Folds are queued such that GPU 0 and GPU 1 each run one fold at a time;
# next fold starts on the GPU that finishes first.

set -euo pipefail

ML_ENV=${ML_ENV:-$HOME/ml-env}
export nnUNet_raw=${nnUNet_raw:-$HOME/nnunet_raw}
export nnUNet_preprocessed=${nnUNet_preprocessed:-$HOME/nnunet_preprocessed}
export nnUNet_results=${nnUNet_results:-$HOME/nnunet_results}

# Numpy 2.x can OOM Masuuni at high thread counts during nnU-Net data loading
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

source "$ML_ENV/bin/activate"

LOG_DIR=$HOME/nnunet_logs
mkdir -p "$LOG_DIR"

preprocess() {
  echo "=== plan_and_preprocess D006 + D007 (3d_fullres, default planner) ==="
  nnUNetv2_plan_and_preprocess \
      -d 6 7 \
      -c 3d_fullres \
      --verify_dataset_integrity \
      2>&1 | tee "$LOG_DIR/preprocess.log"
}

# Two simple per-GPU sequential queues running in parallel.
# GPU 0 owns D006 (5 folds serial), GPU 1 owns D007 (5 folds serial).
# Each queue: ~12 h/fold × 5 = ~60 h wall.
gpu_queue() {
  local gpu=$1 dataset=$2
  shift 2
  local folds=("$@")
  for f in "${folds[@]}"; do
    local log="$LOG_DIR/d${dataset}_fold${f}_gpu${gpu}.log"
    echo "[$(date -Is)] gpu=$gpu d=$dataset fold=$f START" >> "$LOG_DIR/queue_gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES=$gpu \
      nnUNetv2_train "$dataset" 3d_fullres "$f" --npz \
      > "$log" 2>&1
    echo "[$(date -Is)] gpu=$gpu d=$dataset fold=$f DONE exit=$?" >> "$LOG_DIR/queue_gpu${gpu}.log"
  done
}

train_all() {
  gpu_queue 0 6 0 1 2 3 4 &
  local pid0=$!
  gpu_queue 1 7 0 1 2 3 4 &
  local pid1=$!
  echo "[$(date -Is)] launched gpu0 pid=$pid0 (D006 folds 0..4), gpu1 pid=$pid1 (D007 folds 0..4)" \
      | tee "$LOG_DIR/queue_master.log"
  wait "$pid0" "$pid1"
  echo "[$(date -Is)] all 10 folds finished" | tee -a "$LOG_DIR/queue_master.log"
}

case "${1:-}" in
  preprocess) preprocess ;;
  train_all)  train_all ;;
  *) echo "usage: $0 {preprocess|train_all}"; exit 2 ;;
esac
