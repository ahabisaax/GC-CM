#!/bin/bash -l
#$ -N CelebA_Eval_LastCkpt
#$ -o ~/Scratch/xai-crcbm/logs/celeba_eval_last_ckpt_$JOB_ID.out
#$ -e ~/Scratch/xai-crcbm/logs/celeba_eval_last_ckpt_$JOB_ID.err
#$ -pe smp 8
#$ -l h_rt=35:00:00
#$ -l mem=4G
#$ -l tmpfs=20G
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
FINAL_RESULTS_DIR=$PROJECT_ROOT/results
DATASET_TAR="celeba.tar"

echo "--- SETTING UP LOCAL WORKSPACE ---"
LOCAL_WORKSPACE="$TMPDIR/$JOB_ID"
mkdir -p "$LOCAL_WORKSPACE/data"

echo "Copying code..."
rsync -a \
    --exclude '/results' \
    --exclude '/logs' \
    --exclude '/wandb' \
    --exclude '/data' \
    --exclude '.git' \
    "$PROJECT_ROOT/" "$LOCAL_WORKSPACE/"

echo "Copying dataset..."
cp "$PROJECT_ROOT/data/$DATASET_TAR" "$LOCAL_WORKSPACE/data/"
cd "$LOCAL_WORKSPACE/data" && tar -xf $DATASET_TAR

echo "Copying model checkpoints..."
mkdir -p "$LOCAL_WORKSPACE/results"
rsync -a "$PROJECT_ROOT/results/celeba_cem_39c/" "$LOCAL_WORKSPACE/results/celeba_cem_39c/"

echo "Copying existing results dict (if any)..."
RESULTS_DICT="$PROJECT_ROOT/results/results_celeba_suite_last_ckpt.dict"
[ -f "$RESULTS_DICT" ] && cp "$RESULTS_DICT" "$LOCAL_WORKSPACE/results/"

cd "$LOCAL_WORKSPACE"
export PYTHONPATH="$LOCAL_WORKSPACE:$PYTHONPATH"

# --- 4. RUN EVAL ---
echo "--- Starting CelebA last-checkpoint eval ---"
$CONDA_PREFIX/bin/python -u experiments/evaluate_models/evaluate_celeba_last_ckpt.py

# --- 5. SYNC RESULTS BACK ---
echo "Syncing results back..."
rsync -a "$LOCAL_WORKSPACE/results/results_celeba_suite_last_ckpt.dict" \
         "$FINAL_RESULTS_DIR/"
echo "Done."
