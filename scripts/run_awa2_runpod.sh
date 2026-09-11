#!/bin/bash
# RunPod equivalent of experiments/run_awa2_gccem_5fold_worker.sh.
#
#   Usage: bash scripts/run_awa2_runpod.sh [config]
#     config  default experiments/configs/awa2_cem_5fold.yaml
#             pass experiments/configs/awa2_gccem_5fold.yaml for GC-CEM
#
#   Env: NUM_WORKERS  dataloader workers (default: config value, 8).
#                     Throughput only - never affects results. Set from the
#                     container's cgroup CPU quota, not nproc.
#        AUTOSTOP=1   stop this pod when the run finishes (needs runpodctl
#                     configured and RUNPOD_POD_ID set, which RunPod does).
#        RESULTS_DIR  default /workspace/results/awa2_5fold
#
# One process runs all 5 folds, exactly as the single HPC job does. No
# per-seed sharding and no checkpoint_monitor override: the config's own
# early_stopping_monitor (val_y_accuracy) selects the saved checkpoint, and
# eval_from_last=true means reported metrics come from the last epoch.
#
# Writes results to LOCAL disk during the run, then rsyncs to /workspace at
# the end - same reason as the HPC worker's $TMPDIR staging.
set -euo pipefail

CONFIG="${1:-experiments/configs/awa2_cem_5fold.yaml}"
REPO="${REPO:-/workspace/GC-CM}"
VENV="${VENV:-/root/venv}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/results/awa2_5fold}"
LOGS_DIR="${LOGS_DIR:-/workspace/logs}"
LOCAL_RESULTS="${LOCAL_RESULTS:-/root/results_temp}"

cd "$REPO"
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 2; }

# shellcheck disable=SC1091
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

# Non-interactive shells do not read .bashrc, so pick the key up from the
# persistent env file on /workspace.
# shellcheck disable=SC1091
[ -f /workspace/.sweep_env ] && source /workspace/.sweep_env
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    echo "WARNING: no W&B credentials - running offline" >&2
    export WANDB_MODE=offline
else
    export WANDB_MODE="${WANDB_MODE:-online}"
fi

mkdir -p "$LOCAL_RESULTS" "$RESULTS_DIR" "$LOGS_DIR"
TAG="$(basename "$CONFIG" .yaml)"
LOG_FILE="$LOGS_DIR/${TAG}_$(date +%Y%m%d_%H%M%S).log"

WORKER_ARGS=()
[ -n "${NUM_WORKERS:-}" ] && WORKER_ARGS=(-p dataset_config.num_workers "$NUM_WORKERS")

echo "RUN  config=$CONFIG"
echo "     local_out=$LOCAL_RESULTS  final_out=$RESULTS_DIR"
echo "     log=$LOG_FILE"

set +e
python -u experiments/run_experiments.py \
    --config "$CONFIG" \
    --project_name "AwA2" \
    --output_dir "$LOCAL_RESULTS" \
    "${WORKER_ARGS[@]}" \
    2>&1 | tee "$LOG_FILE"
rc=${PIPESTATUS[0]}
set -e

echo "-- syncing results to $RESULTS_DIR"
rsync -a "$LOCAL_RESULTS/" "$RESULTS_DIR/"
if [ -d "$REPO/wandb" ]; then
    mkdir -p /workspace/wandb && rsync -a "$REPO/wandb/" /workspace/wandb/
fi
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) FINISH rc=$rc config=$CONFIG" | tee -a "$LOGS_DIR/runs.log"

if [ "${AUTOSTOP:-0}" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "-- autostop: stopping pod $RUNPOD_POD_ID"
    runpodctl pod stop "$RUNPOD_POD_ID" || true
fi
exit "$rc"
