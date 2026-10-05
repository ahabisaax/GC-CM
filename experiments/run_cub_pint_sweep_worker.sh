#!/bin/bash -l
#$ -N CUB_pint_sweep
#$ -o ~/Scratch/xai-crcbm/logs/CUB_pint_sweep_$JOB_ID.out
#$ -e ~/Scratch/xai-crcbm/logs/CUB_pint_sweep_$JOB_ID.err
#$ -pe smp 8
#$ -l h_rt=24:00:00
#$ -l mem=24G
#$ -l tmpfs=20G
#$ -wd /home/ucakais/Scratch/xai-crcbm
#$ -l gpu=1
#$ -P Gold
#$ -A hpc.28

# CUB p_int sweep at lambda_c = 0.1: CEM and GC-CEM x p_int {0, 0.25, 0.5, 0.75},
# one fold, 100 epochs -> 8 runs, trained one after another in this job.
# Config: experiments/configs/cub_pint_sweep_lam0_1.yaml
#
# Results are written to $TMPDIR for speed and synced to Scratch every 30 min
# and at the end, so finished runs survive if the job hits h_rt. Resubmitting
# reloads finished models (.pt) from Scratch instead of retraining them.

# --- 1. LOAD MODULES ---
module purge
module unload compilers mpi gcc-libs

module load python3/3.9-gnu-10.2.0
module load gcc-libs/10.2.0

export CC=$(which gcc)
export CXX=$(which g++)

echo "Sourcing conda setup..."
conda activate xai2

# --- 2. SET THREADING ---
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${NSLOTS}"
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS

# --- 3. PATHS ---
PROJECT_ROOT=~/Scratch/xai-crcbm
FINAL_RESULTS_DIR=$PROJECT_ROOT/results/cub_pint_sweep_lam0_1
DATASET_TAR="CUB200.tar"

echo "--- SETTING UP FAST LOCAL WORKSPACE ---"
LOCAL_WORKSPACE="$TMPDIR/$JOB_ID"
mkdir -p "$LOCAL_WORKSPACE/data"

echo "Copying code to local SSD..."
rsync -a \
    --exclude '/results' \
    --exclude '/logs' \
    --exclude '/wandb' \
    --exclude '/data' \
    --exclude '.git' \
    "$PROJECT_ROOT/" "$LOCAL_WORKSPACE/"

echo "Copying $DATASET_TAR to local SSD..."
cp "$PROJECT_ROOT/data/$DATASET_TAR" "$LOCAL_WORKSPACE/data/"

cd "$LOCAL_WORKSPACE/data"
tar -xf $DATASET_TAR

# cub_loader.py is source code, not dataset content — copy it explicitly
mkdir -p "$LOCAL_WORKSPACE/data/CUB200"
cp "$PROJECT_ROOT/data/CUB200/"*.py "$LOCAL_WORKSPACE/data/CUB200/"

cd "$LOCAL_WORKSPACE"
export PYTHONPATH="$LOCAL_WORKSPACE:$PYTHONPATH"
echo "Running from: $(pwd)"

LOCAL_CONFIG="experiments/configs/cub_pint_sweep_lam0_1.yaml"
LOCAL_RESULTS="$TMPDIR/results_temp"
mkdir -p "$LOCAL_RESULTS" "$FINAL_RESULTS_DIR"

# Seed local results from Scratch so a resubmitted job reuses finished runs
rsync -a "$FINAL_RESULTS_DIR/" "$LOCAL_RESULTS/"

# Periodic sync back to Scratch, so a job killed at h_rt keeps finished runs
( while true; do sleep 1800; rsync -a "$LOCAL_RESULTS/" "$FINAL_RESULTS_DIR/"; done ) &
SYNC_PID=$!

echo "Starting Training..."
$CONDA_PREFIX/bin/python -u experiments/run_experiments.py \
    --config "$LOCAL_CONFIG" \
    --project_name "CUB_pint_sweep" \
    --output_dir "$LOCAL_RESULTS"

# --- 4. SAVE RESULTS ---
kill $SYNC_PID 2>/dev/null
echo "Syncing results back to Scratch..."
rsync -a "$LOCAL_RESULTS/" "$FINAL_RESULTS_DIR/"
echo "Done. Results in $FINAL_RESULTS_DIR"
