#!/bin/bash -l
#$ -N CUB_Eval
#$ -o ~/Scratch/xai-crcbm/logs/CUB_eval_$JOB_ID.out
#$ -e ~/Scratch/xai-crcbm/logs/CUB_eval_$JOB_ID.err
#$ -pe smp 16
#$ -l h_rt=12:00:00
#$ -l mem=16G
#$ -l tmpfs=20G
#$ -wd /home/ucakais/Scratch/xai-crcbm
#$ -P Gold
#$ -A hpc.28

# No GPU needed — eval is CPU-only (Ridge + logistic probes run in parallel)

# --- 1. LOAD MODULES ---
module purge
module unload compilers mpi gcc-libs
module load python3/3.9-gnu-10.2.0
module load gcc-libs/10.2.0

export CC=$(which gcc)
export CXX=$(which g++)

echo "Sourcing conda setup..."
conda activate xai2

# --- 2. THREADING (give joblib/sklearn all allocated slots) ---
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS
export NUMEXPR_NUM_THREADS=$NSLOTS

# --- 3. PATHS ---
PROJECT_ROOT=~/Scratch/xai-crcbm
CEM_RESULTS_DIR=$PROJECT_ROOT/results/cub_cem
DATASET_TAR="CUB200.tar"

echo "--- SETTING UP FAST LOCAL WORKSPACE ---"
LOCAL_WORKSPACE="$TMPDIR/$JOB_ID"
mkdir -p "$LOCAL_WORKSPACE/data"

# Copy code (no results/logs/wandb — we read checkpoints from Scratch directly)
echo "Copying code to local SSD..."
rsync -a \
    --exclude '/results' \
    --exclude '/logs' \
    --exclude '/wandb' \
    --exclude '/data' \
    --exclude '.git' \
    "$PROJECT_ROOT/" "$LOCAL_WORKSPACE/"

# Copy dataset
echo "Copying $DATASET_TAR to local SSD..."
cp "$PROJECT_ROOT/data/$DATASET_TAR" "$LOCAL_WORKSPACE/data/"
cd "$LOCAL_WORKSPACE/data"
tar -xf "$DATASET_TAR"

# cub_loader.py is source code, not dataset content, and isn't covered by
# CUB200.tar or the code rsync above (which excludes /data) — copy it in
# explicitly.
mkdir -p "$LOCAL_WORKSPACE/data/CUB200"
cp "$PROJECT_ROOT/data/CUB200/"*.py "$LOCAL_WORKSPACE/data/CUB200/"

# Copy trained checkpoints (eval reads from results/cub_cem/)
echo "Copying trained CEM checkpoints to local SSD..."
mkdir -p "$LOCAL_WORKSPACE/results/cub_cem"
rsync -a "$CEM_RESULTS_DIR/" "$LOCAL_WORKSPACE/results/cub_cem/"

# Copy existing eval cache if present (avoids recomputing finished metrics)
if [ -f "$PROJECT_ROOT/results/results_cub_suite.dict" ]; then
    echo "Copying existing results cache..."
    mkdir -p "$LOCAL_WORKSPACE/results"
    cp "$PROJECT_ROOT/results/results_cub_suite.dict" "$LOCAL_WORKSPACE/results/"
fi
if [ -d "$PROJECT_ROOT/results/cache" ]; then
    echo "Copying metric cache..."
    mkdir -p "$LOCAL_WORKSPACE/results/cache"
    rsync -a "$PROJECT_ROOT/results/cache/" "$LOCAL_WORKSPACE/results/cache/"
fi

cd "$LOCAL_WORKSPACE"
export PYTHONPATH="$LOCAL_WORKSPACE:$PYTHONPATH"
echo "Running from: $(pwd)"
echo "NSLOTS=$NSLOTS  OMP_NUM_THREADS=$OMP_NUM_THREADS"

# --- 4. RUN EVAL SUITE ---
echo "Starting CUB evaluation suite..."
$CONDA_PREFIX/bin/python -u experiments/evaluate_models/evaluate_models_cub_suite.py \
    2>&1 | tee "$LOCAL_WORKSPACE/eval_log_$JOB_ID.txt"

# --- 5. SYNC RESULTS BACK ---
echo "Syncing results back to Scratch..."
mkdir -p "$PROJECT_ROOT/results"
mkdir -p "$PROJECT_ROOT/results/plots/cub"
mkdir -p "$PROJECT_ROOT/results/cache"

# Results dict (most important — incremental cache of all metric values)
cp "$LOCAL_WORKSPACE/results/results_cub_suite.dict" \
   "$PROJECT_ROOT/results/results_cub_suite.dict"

# Plots
rsync -a "$LOCAL_WORKSPACE/results/plots/cub/" \
         "$PROJECT_ROOT/results/plots/cub/"

# Metric cache (ICL baseline etc.)
rsync -a "$LOCAL_WORKSPACE/results/cache/" \
         "$PROJECT_ROOT/results/cache/"

# Eval log
cp "$LOCAL_WORKSPACE/eval_log_$JOB_ID.txt" \
   "$PROJECT_ROOT/logs/eval_log_$JOB_ID.txt"

echo "Done. Results at $PROJECT_ROOT/results/results_cub_suite.dict"
