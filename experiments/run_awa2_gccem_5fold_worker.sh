#!/bin/bash -l
#$ -N AwA2_GCCEM_5fold
#$ -o ~/Scratch/xai-crcbm/logs/AwA2_GCCEM_$JOB_ID.out
#$ -e ~/Scratch/xai-crcbm/logs/AwA2_GCCEM_$JOB_ID.err
#$ -pe smp 8
#$ -l h_rt=36:00:00
#$ -l mem=4G
#$ -l tmpfs=40G
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

export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${NSLOTS}"
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS

PROJECT_ROOT=~/Scratch/xai-crcbm
FINAL_RESULTS_DIR=$PROJECT_ROOT/results/awa2_5fold
# Stage AwA2 onto the node's LOCAL scratch ($TMPDIR), not a symlink to Scratch.
# Measured on RunPod: reading these ~348KB JPEGs from network storage is ~82x
# slower than local disk, and training is dataloader-bound either way.
# Extracted AwA2 is ~14GB, so tmpfs must be >= 40G (set above).
# Fetch it first on a LOGIN node:  bash experiments/fetch_awa2_hpc.sh
echo "Staging AwA2 to $LOCAL_WORKSPACE/data (local disk)..."
cp -r "$PROJECT_ROOT/data/AwA2" "$LOCAL_WORKSPACE/data/AwA2"
echo "  staged: $(find "$LOCAL_WORKSPACE/data/AwA2" -name '*.jpg' | wc -l) images"

cd "$LOCAL_WORKSPACE"
export PYTHONPATH="$LOCAL_WORKSPACE:$PYTHONPATH"

LOCAL_CONFIG="experiments/configs/awa2_gccem_5fold.yaml"
LOCAL_RESULTS="$TMPDIR/results_temp"

export WANDB_MODE=online
$CONDA_PREFIX/bin/python -u experiments/run_experiments.py \
    --config "$LOCAL_CONFIG" \
    --project_name "AwA2" \
    --output_dir "$LOCAL_RESULTS"

mkdir -p "$FINAL_RESULTS_DIR"
rsync -a "$LOCAL_RESULTS/" "$FINAL_RESULTS_DIR/"

echo "Syncing wandb offline runs..."
mkdir -p "$PROJECT_ROOT/wandb"
rsync -a "$LOCAL_WORKSPACE/wandb/" "$PROJECT_ROOT/wandb/"
