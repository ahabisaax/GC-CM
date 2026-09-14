#!/bin/bash
# AwA2 CBM arm: 8 jobs x 5 folds, sequential on one pod.
#
#   Usage: bash scripts/run_awa2_cbm_queue.sh
#   Env:   NUM_WORKERS, AUTOSTOP, RESULTS_DIR  (as in run_awa2_queue.sh)
#
# Separate STATE_DIR from the CEM queue so the two resume independently.
#
# Order pairs each lambda before moving on, so if the pod stops early you
# still have a complete joint-vs-GC comparison at the lambdas that finished
# rather than three joints and no GC-CBM. Hard and Seq are last: they are
# fixed-lambda baselines, not part of the lambda sweep.
set -uo pipefail

REPO="${REPO:-/workspace/GC-CM}"
VENV="${VENV:-/root/venv}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/results/awa2_5fold_cbm}"
LOGS_DIR="${LOGS_DIR:-/workspace/logs}"
STATE_DIR="${STATE_DIR:-/workspace/queue_state_cbm}"

cd "$REPO"
# shellcheck disable=SC1091
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
# shellcheck disable=SC1091
[ -f /workspace/.sweep_env ] && source /workspace/.sweep_env
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    echo "WARNING: no W&B credentials - running offline" >&2
    export WANDB_MODE=offline
else
    export WANDB_MODE="${WANDB_MODE:-online}"
fi

mkdir -p "$RESULTS_DIR" "$LOGS_DIR" "$STATE_DIR"

# tag | config | start_split | trials
JOBS=(
  "cbm_lam0.1|experiments/configs/awa2_cbm_5fold_lam0_1.yaml|0|5"
  "gccbm_lam0.1|experiments/configs/awa2_gccbm_5fold_lam0_1.yaml|0|5"
  "cbm_lam0.5|experiments/configs/awa2_cbm_5fold_lam0_5.yaml|0|5"
  "gccbm_lam0.5|experiments/configs/awa2_gccbm_5fold_lam0_5.yaml|0|5"
  "cbm_lam1.0|experiments/configs/awa2_cbm_5fold_lam1_0.yaml|0|5"
  "gccbm_lam1.0|experiments/configs/awa2_gccbm_5fold_lam1_0.yaml|0|5"
  "hardcbm|experiments/configs/awa2_hardcbm_5fold.yaml|0|5"
  "seqcbm|experiments/configs/awa2_seqcbm_5fold.yaml|0|5"
)

echo "=== AwA2 CBM queue: ${#JOBS[@]} jobs ==="
for j in "${JOBS[@]}"; do
    IFS='|' read -r tag cfg start trials <<< "$j"
    printf '  %-16s %s\n' "$tag" "$cfg"
done
echo

for j in "${JOBS[@]}"; do
    IFS='|' read -r tag cfg start trials <<< "$j"
    sentinel="$STATE_DIR/.done_$tag"
    if [ -f "$sentinel" ]; then echo "SKIP $tag"; continue; fi
    if [ ! -f "$cfg" ]; then
        echo "FAIL $tag: config not found: $cfg" | tee -a "$LOGS_DIR/queue_cbm.log"; continue
    fi
    log="$LOGS_DIR/${tag}_$(date +%Y%m%d_%H%M%S).log"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) START $tag cfg=$cfg" | tee -a "$LOGS_DIR/queue_cbm.log"

    python -u experiments/run_experiments.py \
        --config "$cfg" \
        --project_name "AwA2" \
        --output_dir "$RESULTS_DIR" \
        -p start_split "$start" \
        -p trials "$trials" \
        -p dataset_config.num_workers "${NUM_WORKERS:-8}" \
        > "$log" 2>&1
    rc=$?

    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) FINISH $tag rc=$rc log=$log" | tee -a "$LOGS_DIR/queue_cbm.log"
    if [ "$rc" -eq 0 ]; then touch "$sentinel"; else
        echo "  ^ nonzero rc, no sentinel; queue continues" | tee -a "$LOGS_DIR/queue_cbm.log"
        tail -20 "$log" | sed 's/^/  | /'
    fi
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) CBM QUEUE COMPLETE" | tee -a "$LOGS_DIR/queue_cbm.log"
if [ "${AUTOSTOP:-0}" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    runpodctl pod stop "$RUNPOD_POD_ID" || true
fi
