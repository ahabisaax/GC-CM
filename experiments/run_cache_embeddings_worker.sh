#!/bin/bash -l
#$ -N Cache_Embeddings
#$ -o ~/Scratch/xai-crcbm/logs/cache_embeddings_$JOB_ID.out
#$ -e ~/Scratch/xai-crcbm/logs/cache_embeddings_$JOB_ID.err
#$ -pe smp 8
#$ -l h_rt=4:00:00
#$ -l mem=8G
#$ -l tmpfs=30G
#$ -wd /home/ucakais/Scratch/xai-crcbm
#$ -l gpu=1
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
mkdir -p "$LOCAL_WORKSPACE/data"

echo "Copying code..."
rsync -a \
    --exclude '/results' \
    --exclude '/logs' \
    --exclude '/wandb' \
    --exclude '/data' \
    --exclude '.git' \
    "$PROJECT_ROOT/" "$LOCAL_WORKSPACE/"

echo "Copying data..."
rsync -a "$PROJECT_ROOT/data/" "$LOCAL_WORKSPACE/data/"

echo "Copying existing results (checkpoints)..."
rsync -a "$PROJECT_ROOT/results/" "$LOCAL_WORKSPACE/results/"

cd "$LOCAL_WORKSPACE"
export PYTHONPATH="$LOCAL_WORKSPACE:$PYTHONPATH"
echo "Running from: $(pwd)"

echo ""
echo "=== Caching TabularToy embeddings ==="
$CONDA_PREFIX/bin/python -u experiments/evaluate_models/cache_embeddings_all.py \
    --dataset tabulartoy

echo ""
echo "=== Caching dSprites embeddings ==="
$CONDA_PREFIX/bin/python -u experiments/evaluate_models/cache_embeddings_all.py \
    --dataset dsprites

echo ""
echo "=== Caching CUB embeddings ==="
$CONDA_PREFIX/bin/python -u experiments/evaluate_models/cache_embeddings_all.py \
    --dataset cub

echo ""
echo "--- Syncing embedding caches back to Scratch ---"
rsync -a "$LOCAL_WORKSPACE/results/embeddings_tabulartoy_all.joblib" "$PROJECT_ROOT/results/"
rsync -a "$LOCAL_WORKSPACE/results/embeddings_dsprites_all.joblib"   "$PROJECT_ROOT/results/"
rsync -a "$LOCAL_WORKSPACE/results/embeddings_cub_all.joblib"        "$PROJECT_ROOT/results/"

echo "Done."
