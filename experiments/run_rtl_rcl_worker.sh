#!/bin/bash -l
#$ -N RTL_RCL_All_Datasets
#$ -o ~/Scratch/xai-crcbm/logs/rtl_rcl_$JOB_ID.out
#$ -e ~/Scratch/xai-crcbm/logs/rtl_rcl_$JOB_ID.err
#$ -pe smp 8
#$ -l h_rt=12:00:00
#$ -l mem=16G
#$ -l tmpfs=10G
#$ -wd /home/ucakais/Scratch/xai-crcbm
#$ -P Gold
#$ -A hpc.28

module purge
module unload compilers mpi gcc-libs
module load python3/3.9-gnu-10.2.0
module load gcc-libs/10.2.0

export CC=$(which gcc)
export CXX=$(which g++)

conda activate xai2

export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS

PROJECT_ROOT=~/Scratch/xai-crcbm

echo "--- SETTING UP LOCAL WORKSPACE ---"
LOCAL_WORKSPACE="$TMPDIR/$JOB_ID"
mkdir -p "$LOCAL_WORKSPACE/results"

echo "Copying code..."
rsync -a \
    --exclude '/results' \
    --exclude '/logs' \
    --exclude '/wandb' \
    --exclude '/data' \
    --exclude '.git' \
    "$PROJECT_ROOT/" "$LOCAL_WORKSPACE/"

echo "Copying embedding caches..."
rsync -a "$PROJECT_ROOT/results/embeddings_tabulartoy_all.joblib" "$LOCAL_WORKSPACE/results/"
rsync -a "$PROJECT_ROOT/results/embeddings_dsprites_all.joblib"   "$LOCAL_WORKSPACE/results/"
rsync -a "$PROJECT_ROOT/results/embeddings_cub_all.joblib"        "$LOCAL_WORKSPACE/results/"

# Copy existing results dict if present (enables resume)
[ -f "$PROJECT_ROOT/results/results_rtl_rcl_all_datasets.dict" ] && \
    rsync -a "$PROJECT_ROOT/results/results_rtl_rcl_all_datasets.dict" "$LOCAL_WORKSPACE/results/"

cd "$LOCAL_WORKSPACE"
export PYTHONPATH="$LOCAL_WORKSPACE:$PYTHONPATH"
echo "Running from: $(pwd)"

echo ""
echo "=== Computing RTL / RCL (Ridge + MLP) — all datasets ==="
$CONDA_PREFIX/bin/python -u experiments/evaluate_models/compute_rtl_rcl_all_datasets.py \
    --datasets tabulartoy dsprites cub

echo ""
echo "--- Syncing results back to Scratch ---"
rsync -a "$LOCAL_WORKSPACE/results/results_rtl_rcl_all_datasets.dict" "$PROJECT_ROOT/results/"

echo "Done."
