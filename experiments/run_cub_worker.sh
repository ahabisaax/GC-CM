#!/bin/bash -l
#$ -N CBM_Leakage_Run
#$ -o ~/Scratch/xai-crcbm/logs/CUB_$JOB_ID.out
#$ -e ~/Scratch/xai-crcbm/logs/test_run_$JOB_ID.err
#$ -pe smp 16
#$ -l h_rt=24:00:00
#$ -l mem=16G
#$ -l tmpfs=20G
#$ -wd /home/ucakais/Scratch/xai-crcbm
#$ -l gpu=8

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
#CHANGE FOR EACH DATASET
FINAL_RESULTS_DIR=$PROJECT_ROOT/results/cub_soft_0.01
DATASET_TAR="CUB200.tar"


echo "--- SETTING UP FAST LOCAL WORKSPACE ---"
LOCAL_WORKSPACE="$TMPDIR/$JOB_ID"
mkdir -p "$LOCAL_WORKSPACE/data"

echo "Copying code to local SSD..."
cd "$LOCAL_WORKSPACE/data"
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

cd "$LOCAL_WORKSPACE"
export PYTHONPATH="$LOCAL_WORKSPACE:$PYTHONPATH"
echo "Running from: $(pwd)"


LOCAL_CONFIG="experiments/configs/cub.yaml"
LOCAL_RESULTS="$TMPDIR/results_temp"

echo "Starting Training..."
$CONDA_PREFIX/bin/python -u experiments/run_experiments.py \
    --config "$LOCAL_CONFIG" \
    --project_name "CUB_Fast" \
    --output_dir "$LOCAL_RESULTS"

# --- 6. SAVE RESULTS ---
echo "Syncing results back to Scratch..."

mkdir -p "$FINAL_RESULTS_DIR"
rsync -a "$LOCAL_RESULTS/" "$FINAL_RESULTS_DIR/"
