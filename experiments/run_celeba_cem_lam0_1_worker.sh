#!/bin/bash -l
#$ -N CelebA_CEM_lam0_1
#$ -o ~/Scratch/xai-crcbm/logs/CelebA_CEM_lam0_1_$JOB_ID.out
#$ -e ~/Scratch/xai-crcbm/logs/CelebA_CEM_lam0_1_$JOB_ID.err
#$ -pe smp 8
#$ -l h_rt=30:30:00
#$ -l mem=24G
#$ -l tmpfs=20G
#$ -wd /home/ucakais/Scratch/xai-crcbm
#$ -l gpu=1

module purge
module unload compilers mpi gcc-libs
module load python3/3.9-gnu-10.2.0
module load gcc-libs/10.2.0

export CC=$(which gcc)
export CXX=$(which g++)

conda activate xai2

export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${NSLOTS}"
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS

PROJECT_ROOT=~/Scratch/xai-crcbm
FINAL_RESULTS_DIR=$PROJECT_ROOT/results/celeba_cem
DATASET_TAR="celeba.tar"

LOCAL_WORKSPACE="$TMPDIR/$JOB_ID"
mkdir -p "$LOCAL_WORKSPACE/data"

rsync -a \
    --exclude '/results' --exclude '/logs' --exclude '/wandb' \
    --exclude '/data' --exclude '.git' \
    "$PROJECT_ROOT/" "$LOCAL_WORKSPACE/"

cp "$PROJECT_ROOT/data/$DATASET_TAR" "$LOCAL_WORKSPACE/data/"
cd "$LOCAL_WORKSPACE/data" && tar -xf $DATASET_TAR

cd "$LOCAL_WORKSPACE"
export PYTHONPATH="$LOCAL_WORKSPACE:$PYTHONPATH"

LOCAL_CONFIG="experiments/configs/celeba_cem.yaml"
LOCAL_RESULTS="$TMPDIR/results_temp"

$CONDA_PREFIX/bin/python -u experiments/run_experiments.py \
    --config "$LOCAL_CONFIG" \
    --project_name "CelebA_CEM" \
    --output_dir "$LOCAL_RESULTS" \
    --filter_in "^CEM.*lam_c0.1"

mkdir -p "$FINAL_RESULTS_DIR"
rsync -a "$LOCAL_RESULTS/" "$FINAL_RESULTS_DIR/"
