#!/bin/bash
# Sequential AwA2 job queue for a single pod.
#
#   Usage: bash scripts/run_awa2_queue.sh
#   Env:   NUM_WORKERS  dataloader workers (default 8; throughput only)
#          AUTOSTOP=1   stop the pod after the LAST job (needs runpodctl
#                       configured on the pod and RUNPOD_POD_ID set)
#          RESULTS_DIR  default /workspace/results/awa2_5fold
#
# Jobs run one at a time, in order. Each writes a sentinel on success, so
# re-running the script after an interruption resumes at the first unfinished
# job rather than redoing completed ones. A failing job does not stop the
# queue: its rc is recorded and the next job starts, because a crash in one
# lambda should not cost the remaining ones (this is exactly how the GC-CEM
# lam 0.1 run lost folds 4 and 5 on the cluster).
#
# Job 1 resumes GC-CEM lam_c 0.1 at split 2. run_experiments.py iterates
# range(start_split, trials), so start_split=2 trials=5 runs splits 2,3,4,
# which W&B names fold_3, fold_4, fold_5 - the three the cluster never
# produced. Folds 1 and 2 are already finished and are not touched.
set -uo pipefail

REPO="${REPO:-/workspace/GC-CM}"
VENV="${VENV:-/root/venv}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/results/awa2_5fold}"
LOGS_DIR="${LOGS_DIR:-/workspace/logs}"
STATE_DIR="${STATE_DIR:-/workspace/queue_state}"

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
  "gccem_lam0.1_folds3to5|experiments/configs/awa2_gccem_5fold.yaml|2|5"
  "gccem_lam0.5|experiments/configs/awa2_gccem_5fold_lam0_5.yaml|0|5"
  "gccem_lam1.0|experiments/configs/awa2_gccem_5fold_lam1_0.yaml|0|5"
  "cem_lam0.5|experiments/configs/awa2_cem_5fold_lam0_5.yaml|0|5"
  "cem_lam1.0|experiments/configs/awa2_cem_5fold_lam1_0.yaml|0|5"
)

echo "=== AwA2 queue: ${#JOBS[@]} jobs ==="
for j in "${JOBS[@]}"; do
    IFS='|' read -r tag cfg start trials <<< "$j"
    printf '  %-24s %s  splits %s..%s\n' "$tag" "$cfg" "$start" "$((trials-1))"
done
echo

for j in "${JOBS[@]}"; do
    IFS='|' read -r tag cfg start trials <<< "$j"
    sentinel="$STATE_DIR/.done_$tag"
    if [ -f "$sentinel" ]; then
        echo "SKIP $tag (sentinel $sentinel)"
        continue
    fi
    if [ ! -f "$cfg" ]; then
        echo "FAIL $tag: config not found: $cfg" | tee -a "$LOGS_DIR/queue.log"
        continue
    fi
    log="$LOGS_DIR/${tag}_$(date +%Y%m%d_%H%M%S).log"
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) START $tag cfg=$cfg splits=$start..$((trials-1))" \
        | tee -a "$LOGS_DIR/queue.log"

    python -u experiments/run_experiments.py \
        --config "$cfg" \
        --project_name "AwA2" \
        --output_dir "$RESULTS_DIR" \
        -p start_split "$start" \
        -p trials "$trials" \
        -p dataset_config.num_workers "${NUM_WORKERS:-8}" \
        > "$log" 2>&1
    rc=$?

    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) FINISH $tag rc=$rc log=$log" \
        | tee -a "$LOGS_DIR/queue.log"
    if [ "$rc" -eq 0 ]; then
        touch "$sentinel"
    else
        echo "  ^ nonzero rc, no sentinel written; queue continues" | tee -a "$LOGS_DIR/queue.log"
        tail -20 "$log" | sed 's/^/  | /'
    fi
done

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) QUEUE COMPLETE" | tee -a "$LOGS_DIR/queue.log"
if [ -d "$REPO/wandb" ]; then
    mkdir -p /workspace/wandb && rsync -a "$REPO/wandb/" /workspace/wandb/ 2>/dev/null || true
fi
if [ "${AUTOSTOP:-0}" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
    echo "-- autostop: stopping pod $RUNPOD_POD_ID"
    runpodctl pod stop "$RUNPOD_POD_ID" || true
fi
