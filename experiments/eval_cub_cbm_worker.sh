#!/bin/bash -l
#$ -N CUB_CBM_Eval
#$ -o ~/Scratch/xai-crcbm/logs/CUB_CBM_eval_$JOB_ID.out
#$ -e ~/Scratch/xai-crcbm/logs/CUB_CBM_eval_$JOB_ID.err
#$ -pe smp 16
#$ -l h_rt=12:00:00
#$ -l mem=16G
#$ -l tmpfs=20G
#$ -wd /home/ucakais/Scratch/xai-crcbm

module purge
module unload compilers mpi gcc-libs
module load python3/3.9-gnu-10.2.0
module load gcc-libs/10.2.0

export CC=$(which gcc)
export CXX=$(which g++)

echo "Activating conda env xai-leakage..."
conda activate xai-leakage

export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS
export NUMEXPR_NUM_THREADS=$NSLOTS

PROJECT_ROOT=~/Scratch/xai-crcbm

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

echo "Copying CUB200 dataset to local SSD..."
cp "$PROJECT_ROOT/data/CUB200.tar" "$LOCAL_WORKSPACE/data/"
cd "$LOCAL_WORKSPACE/data" && tar -xf CUB200.tar && cd "$LOCAL_WORKSPACE"

echo "Copying CBM checkpoints to local SSD..."
mkdir -p "$LOCAL_WORKSPACE/results/cub_soft_0.01"
mkdir -p "$LOCAL_WORKSPACE/results/cub_acbm_shared_critic"
rsync -a "$PROJECT_ROOT/results/cub_soft_0.01/"         "$LOCAL_WORKSPACE/results/cub_soft_0.01/"
rsync -a "$PROJECT_ROOT/results/cub_acbm_shared_critic/" "$LOCAL_WORKSPACE/results/cub_acbm_shared_critic/"

if [ -f "$PROJECT_ROOT/results/results_cub_cbm_suite.dict" ]; then
    echo "Copying existing results cache..."
    cp "$PROJECT_ROOT/results/results_cub_cbm_suite.dict" "$LOCAL_WORKSPACE/results/"
fi

cd "$LOCAL_WORKSPACE"
export PYTHONPATH="$LOCAL_WORKSPACE:$PYTHONPATH"
echo "Running from: $(pwd)  NSLOTS=$NSLOTS"

echo "Starting CUB CBM evaluation suite..."
$CONDA_PREFIX/bin/python -u experiments/evaluate_models/evaluate_models_cub_cbm_suite.py \
    2>&1 | tee "$LOCAL_WORKSPACE/cub_cbm_eval_log_$JOB_ID.txt"

echo "--- SYNCING RESULTS BACK ---"
mkdir -p "$PROJECT_ROOT/results" "$PROJECT_ROOT/logs"

cp "$LOCAL_WORKSPACE/results/results_cub_cbm_suite.dict" \
   "$PROJECT_ROOT/results/results_cub_cbm_suite.dict"

cp "$LOCAL_WORKSPACE/cub_cbm_eval_log_$JOB_ID.txt" \
   "$PROJECT_ROOT/logs/cub_cbm_eval_$JOB_ID.txt"

echo "Done. Results at $PROJECT_ROOT/results/results_cub_cbm_suite.dict"
