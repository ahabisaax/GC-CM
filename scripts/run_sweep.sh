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

MODELS=(cem gc_cem)
LAMBDAS=(0.1 0.5 1.0)
SEEDS=(0 1 2 3 4)

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
    exit 1
fi
echo "all 30 runs done (or already complete)"
