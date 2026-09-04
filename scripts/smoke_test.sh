#!/bin/bash
# Smoke test: TabularToy for 2 epochs + CUB for 1 epoch, one seed each.
# Prints peak GPU memory (torch.cuda.max_memory_allocated) after each run.
# Exits non-zero on any failure.
#
# W&B runs in offline mode here regardless of WANDB_API_KEY — a smoke test
# should not pollute the real project.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
RESULTS_DIR="${RESULTS_DIR:-$WORKSPACE/results}"
LOGS_DIR="${LOGS_DIR:-$WORKSPACE/logs}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

export WANDB_MODE=offline
SMOKE_OUT="$RESULTS_DIR/smoke"
mkdir -p "$SMOKE_OUT" "$LOGS_DIR"

# TabularToy data is generated if missing (fast, no download)
if [ ! -d "data/TabularToy/tabulartoy_25_10k" ]; then
    echo "-- generating TabularToy data for smoke test"
    python data/generate_tabulartoy_dataset.py 0.25 10000
fi

echo "===================================================="
echo "SMOKE 1/2: TabularToy, 2 epochs, 1 seed"
echo "===================================================="
python scripts/smoke_test.py \
    --config experiments/configs/tabulartoy.yaml \
    --output_dir "$SMOKE_OUT/tabulartoy" \
    --filter_in "lam_c1" \
    -p trials 1 \
    -p max_epochs 2 \
    -p check_val_every_n_epoch 1 \
    2>&1 | tee "$LOGS_DIR/smoke_tabulartoy.log"

echo "===================================================="
echo "SMOKE 2/2: CUB, 1 epoch, 1 seed"
echo "===================================================="
if [ -z "$(ls -A "data/CUB200/class_attr_data_10" 2>/dev/null)" ]; then
    echo "-- CUB data not present (celeba-only setup?) — skipping CUB smoke test"
else
    python scripts/smoke_test.py \
        --config experiments/configs/cub.yaml \
        --output_dir "$SMOKE_OUT/cub" \
        -p trials 1 \
        -p max_epochs 1 \
        -p check_val_every_n_epoch 1 \
        2>&1 | tee "$LOGS_DIR/smoke_cub.log"
fi

echo "===================================================="
echo "SMOKE TEST PASSED"
echo "===================================================="
