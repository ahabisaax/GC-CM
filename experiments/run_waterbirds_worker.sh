#!/bin/bash -l
#$ -N CBM_Leakage_Waterbirds
#$ -o ~/Scratch/xai-crcbm/logs/waterbirds_$JOB_ID.out
#$ -e ~/Scratch/xai-crcbm/logs/waterbirds_$JOB_ID.err
#$ -pe smp 8
#$ -l h_rt=48:00:00
#$ -l mem=4G
#$ -l tmpfs=30G
#$ -wd /home/ucakais/Scratch/xai-crcbm
#$ -l gpu=1
#$ -P Gold
#$ -A hpc.28

# --- 1. LOAD MODULES ---
module purge
module unload compilers mpi gcc-libs
module load python3/3.9-gnu-10.2.0
module load gcc-libs/10.2.0

export CC=$(which gcc)
export CXX=$(which g++)

echo "Activating conda..."
conda activate xai2

# --- 2. THREADING ---
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${NSLOTS}"
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS

# --- 3. PATHS ---
PROJECT_ROOT=~/Scratch/xai-crcbm
FINAL_RESULTS_DIR=$PROJECT_ROOT/results/waterbirds
WATERBIRDS_TAR="waterbirds.tar"      # pre-packed from waterbird_complete95_forest2water2/
CUB_DATA_DIR=$PROJECT_ROOT/data/CUB200

echo "--- SETTING UP LOCAL WORKSPACE ---"
LOCAL_WORKSPACE="$TMPDIR/$JOB_ID"
mkdir -p "$LOCAL_WORKSPACE/data/CUB200"

echo "Copying code..."
rsync -a \
    --exclude '/results' \
    --exclude '/logs' \
    --exclude '/wandb' \
    --exclude '/data' \
    --exclude '.git' \
    "$PROJECT_ROOT/" "$LOCAL_WORKSPACE/"

echo "Copying Waterbirds dataset..."
cp "$PROJECT_ROOT/data/$WATERBIRDS_TAR" "$LOCAL_WORKSPACE/data/"
cd "$LOCAL_WORKSPACE/data" && tar -xf $WATERBIRDS_TAR
# Rename extracted folder to 'waterbirds' if needed
[ -d "waterbird_complete95_forest2water2" ] && mv waterbird_complete95_forest2water2 waterbirds

echo "Copying CUB200 attributes..."
rsync -a "$CUB_DATA_DIR/class_attr_data_10/" "$LOCAL_WORKSPACE/data/CUB200/class_attr_data_10/"

cd "$LOCAL_WORKSPACE"
export PYTHONPATH="$LOCAL_WORKSPACE:$PYTHONPATH"

LOCAL_CONFIG="experiments/configs/waterbirds.yaml"
LOCAL_RESULTS="$TMPDIR/results_temp"

echo "--- Starting Waterbirds training ---"
$CONDA_PREFIX/bin/python -u experiments/run_experiments.py \
    --config "$LOCAL_CONFIG" \
    --project_name "Waterbirds" \
    --output_dir "$LOCAL_RESULTS"

# --- 4. SYNC RESULTS BACK ---
echo "Syncing results back..."
mkdir -p "$FINAL_RESULTS_DIR"
rsync -a "$LOCAL_RESULTS/" "$FINAL_RESULTS_DIR/"
echo "Done."
