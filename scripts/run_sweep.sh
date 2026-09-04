#!/bin/bash
# Full sweep for one dataset: {cem, gc_cem} x lambda_c {0.1, 0.5, 1.0} x seeds 0-4.
#
#   Usage: run_sweep.sh {dataset}
#
# Resumable: run_one.sh skips any run whose success sentinel already exists,
# so a preempted spot pod can simply relaunch this script and it continues
# from the first incomplete run. A failed run does not abort the sweep; the
# script reports all failures at the end and exits non-zero if there were any.
set -uo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 {dataset}" >&2
    exit 2
fi
DATASET="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Overridable to split a sweep across pods, e.g.:
#   MODELS=cem bash scripts/run_sweep.sh celeba        (pod 1)
#   MODELS=gc_cem bash scripts/run_sweep.sh celeba     (pod 2)
read -ra MODELS <<< "${MODELS:-cem gc_cem}"
read -ra LAMBDAS <<< "${LAMBDAS:-0.1 0.5 1.0}"
read -ra SEEDS <<< "${SEEDS:-0 1 2 3 4}"

failures=()
for model in "${MODELS[@]}"; do
    for lam in "${LAMBDAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            if ! bash "$SCRIPT_DIR/run_one.sh" "$model" "$DATASET" "$lam" "$seed"; then
                failures+=("$model lam=$lam seed=$seed")
                echo "FAIL $model $DATASET lam=$lam seed=$seed (continuing sweep)" >&2
            fi
        done
    done
done

echo "== sweep complete: $DATASET =="
if [ "${#failures[@]}" -gt 0 ]; then
    echo "FAILED RUNS (${#failures[@]}):" >&2
    printf '  %s\n' "${failures[@]}" >&2
fi

# ------------------------------------------------------------------ autostop
# Opt-in: AUTOSTOP=1 stops this pod once the sweep finishes, so an unattended
# run does not idle-bill after the last job. Results live on the network
# volume and survive the stop; restart the pod to resume.
# Stops on success AND on partial failure — a half-finished sweep idling is
# the exact case this protects against. run_sweep.sh is resumable, so
# restarting the pod and rerunning picks up where it left off.
if [ "${AUTOSTOP:-0}" = "1" ]; then
    POD_ID="${RUNPOD_POD_ID:-}"
    if [ -z "$POD_ID" ]; then
        echo "AUTOSTOP set but RUNPOD_POD_ID is empty — not stopping" >&2
    else
        echo "AUTOSTOP: stopping pod $POD_ID in 60s (Ctrl-C to cancel)"
        sleep 60
        if command -v runpodctl >/dev/null 2>&1; then
            runpodctl stop pod "$POD_ID" || \
                echo "runpodctl stop failed — stop the pod manually" >&2
        else
            echo "runpodctl not found — stop the pod manually" >&2
        fi
    fi
fi

if [ "${#failures[@]}" -gt 0 ]; then
    exit 1
fi
echo "all runs done (or already complete)"
