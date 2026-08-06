#!/bin/bash -l
# Local dry-run of the CelebA HPC job (mirrors experiments/run_celeba_worker.sh),
# so packaging/import/config bugs get caught here instead of burning an HPC
# job — this is exactly the class of bug that broke the last two CUB submissions
# (source files silently missing from what git actually has tracked).
#
# What it reproduces, in the same order as run_celeba_worker.sh:
#   1. A clean checkout of the current git HEAD (via `git archive`) — this is
#      what the HPC's ~/Scratch/xai-crcbm actually contains after `git pull`,
#      i.e. only committed files. Anything untracked, uncommitted, or
#      gitignored will NOT be here, same as it wouldn't be on the HPC.
#   2. The exact same rsync excludes (--exclude '/results' '/logs' '/wandb'
#      '/data' '.git') copying that clean checkout into a job workspace.
#   3. Copying + extracting data/celeba.tar the same way.
#   4. Running experiments/run_experiments.py with the same config
#      (celeba_cem.yaml), just capped to 1 epoch / 1 architecture so it
#      finishes in minutes instead of days.
#
# This validates that imports resolve, the config parses, the dataset loads,
# and one training step + checkpoint save completes — i.e. that the actual
# HPC submission won't die the way the last two did. It does NOT validate
# model convergence (that's what the real multi-hundred-epoch HPC run is for).
#
# Usage:
#   bash experiments/local_smoketest_celeba.sh
#
# Override the conda env if needed:
#   CONDA_ENV=myenv bash experiments/local_smoketest_celeba.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-xai-leakage}"

if [ ! -f "$PROJECT_ROOT/data/celeba.tar" ]; then
    echo "ERROR: $PROJECT_ROOT/data/celeba.tar not found — can't simulate the HPC dataset copy step." >&2
    exit 1
fi

LOCAL_WORKSPACE="$(mktemp -d /tmp/celeba_smoketest.XXXXXX)"
echo "Workspace: $LOCAL_WORKSPACE  (~4GB; not auto-deleted, see cleanup note at the end)"

echo
echo "--- 1. Clean checkout of git HEAD (== what HPC has after git pull) ---"
mkdir -p "$LOCAL_WORKSPACE/clean_repo"
git -C "$PROJECT_ROOT" archive HEAD | tar -x -C "$LOCAL_WORKSPACE/clean_repo"

echo
echo "--- 2. rsync with the exact excludes run_celeba_worker.sh uses ---"
mkdir -p "$LOCAL_WORKSPACE/job/data"
rsync -a \
    --exclude '/results' \
    --exclude '/logs' \
    --exclude '/wandb' \
    --exclude '/data' \
    --exclude '.git' \
    "$LOCAL_WORKSPACE/clean_repo/" "$LOCAL_WORKSPACE/job/"

echo
echo "--- 3. Copy + extract celeba.tar (same as worker script) ---"
cp "$PROJECT_ROOT/data/celeba.tar" "$LOCAL_WORKSPACE/job/data/"
( cd "$LOCAL_WORKSPACE/job/data" && tar -xf celeba.tar )

echo
echo "--- 4. Run training exactly as run_celeba_worker.sh does, capped to 1 epoch / CEM only ---"
cd "$LOCAL_WORKSPACE/job"
export PYTHONPATH="$LOCAL_WORKSPACE/job:${PYTHONPATH:-}"
LOCAL_RESULTS="$LOCAL_WORKSPACE/results_temp"

# --num_workers only overrides the top-level shared_params.num_workers, not
# the nested dataset_config.num_workers that celeba_loader.generate_data()
# actually reads — set both explicitly. Forcing 0 workers here also sidesteps
# a macOS-only artifact: PyTorch DataLoader workers use `spawn` by default on
# macOS (requires pickling everything, including local closures) but `fork`
# on Linux/HPC (memory-copies the process, no pickling needed), so a
# multiprocessing worker count > 0 can fail locally on a lambda inside
# celeba_loader.generate_data() that would be fine on the actual HPC job.
conda run -n "$CONDA_ENV" python -u experiments/run_experiments.py \
    --config experiments/configs/celeba_cem.yaml \
    --project_name "CelebA_smoketest" \
    --output_dir "$LOCAL_RESULTS" \
    --num_workers 0 \
    --filter_in "^CEM_" \
    -p max_epochs 1 \
    -p check_val_every_n_epoch 1 \
    -p dataset_config.num_workers 0

echo
echo "=== SMOKE TEST PASSED ==="
echo "Imports, config parsing, dataset loading, and one training step all ran cleanly."
echo "Results (if any): $LOCAL_RESULTS"
echo
echo "Cleanup: rm -rf $LOCAL_WORKSPACE"
