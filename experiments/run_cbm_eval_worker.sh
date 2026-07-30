#!/bin/bash -l
#$ -N CBM_Eval
#$ -o ~/Scratch/xai-crcbm/logs/CBM_eval_$JOB_ID.out
#$ -e ~/Scratch/xai-crcbm/logs/CBM_eval_$JOB_ID.err
#$ -pe smp 16
#$ -l h_rt=8:00:00
#$ -l mem=16G
#$ -l tmpfs=20G
#$ -wd /home/ucakais/Scratch/xai-crcbm
# Pass -v DATASET=tabulartoy  or  -v DATASET=dsprites  or  -v DATASET=cub
# e.g.  qsub -v DATASET=cub experiments/run_cbm_eval_worker.sh

DATASET=${DATASET:-tabulartoy}

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

echo "--- SETTING UP FAST LOCAL WORKSPACE (dataset=$DATASET) ---"
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

if [ "$DATASET" = "tabulartoy" ]; then
    echo "Copying TabularToy data and checkpoints..."
    mkdir -p "$LOCAL_WORKSPACE/data/TabularToy"
    rsync -a "$PROJECT_ROOT/data/TabularToy/" "$LOCAL_WORKSPACE/data/TabularToy/"
    mkdir -p "$LOCAL_WORKSPACE/results/tabulartoy_25_10k_models"
    mkdir -p "$LOCAL_WORKSPACE/results/tabulartoy_25_10k_models_acbm_shared_critic"
    rsync -a "$PROJECT_ROOT/results/tabulartoy_25_10k_models/" \
             "$LOCAL_WORKSPACE/results/tabulartoy_25_10k_models/"
    rsync -a "$PROJECT_ROOT/results/tabulartoy_25_10k_models_acbm_shared_critic/" \
             "$LOCAL_WORKSPACE/results/tabulartoy_25_10k_models_acbm_shared_critic/"
    EVAL_SCRIPT="experiments/evaluate_models/evaluate_models_tabulartoy_cbm_suite.py"
    RESULT_FILE="results/results_tabulartoy_cbm_suite.dict"

elif [ "$DATASET" = "dsprites" ]; then
    echo "Copying dSprites data and checkpoints..."
    mkdir -p "$LOCAL_WORKSPACE/data/dsprites"
    rsync -a "$PROJECT_ROOT/data/dsprites/" "$LOCAL_WORKSPACE/data/dsprites/"
    mkdir -p "$LOCAL_WORKSPACE/results/dsprites_dep_0_models"
    mkdir -p "$LOCAL_WORKSPACE/results/dsprites_sequential_acbm"
    rsync -a "$PROJECT_ROOT/results/dsprites_dep_0_models/" \
             "$LOCAL_WORKSPACE/results/dsprites_dep_0_models/"
    rsync -a "$PROJECT_ROOT/results/dsprites_sequential_acbm/" \
             "$LOCAL_WORKSPACE/results/dsprites_sequential_acbm/"
    EVAL_SCRIPT="experiments/evaluate_models/evaluate_models_dsprites_cbm_suite.py"
    RESULT_FILE="results/results_dsprites_cbm_suite.dict"

elif [ "$DATASET" = "cub" ]; then
    echo "Copying CUB data and checkpoints..."
    cp "$PROJECT_ROOT/data/CUB200.tar" "$LOCAL_WORKSPACE/data/"
    cd "$LOCAL_WORKSPACE/data" && tar -xf CUB200.tar && cd "$LOCAL_WORKSPACE"
    mkdir -p "$LOCAL_WORKSPACE/results/cub_soft_0.01"
    mkdir -p "$LOCAL_WORKSPACE/results/cub_acbm_shared_critic"
    rsync -a "$PROJECT_ROOT/results/cub_soft_0.01/" \
             "$LOCAL_WORKSPACE/results/cub_soft_0.01/"
    rsync -a "$PROJECT_ROOT/results/cub_acbm_shared_critic/" \
             "$LOCAL_WORKSPACE/results/cub_acbm_shared_critic/"
    EVAL_SCRIPT="experiments/evaluate_models/evaluate_models_cub_cbm_suite.py"
    RESULT_FILE="results/results_cub_cbm_suite.dict"

else
    echo "ERROR: unknown DATASET=$DATASET  (use tabulartoy / dsprites / cub)"
    exit 1
fi

# Copy existing results cache if present (incremental — skips completed folds)
if [ -f "$PROJECT_ROOT/$RESULT_FILE" ]; then
    echo "Copying existing results cache..."
    mkdir -p "$LOCAL_WORKSPACE/results"
    cp "$PROJECT_ROOT/$RESULT_FILE" "$LOCAL_WORKSPACE/$RESULT_FILE"
fi

cd "$LOCAL_WORKSPACE"
export PYTHONPATH="$LOCAL_WORKSPACE:$PYTHONPATH"
echo "Running from: $(pwd)  NSLOTS=$NSLOTS  DATASET=$DATASET"

$CONDA_PREFIX/bin/python -u "$EVAL_SCRIPT" \
    2>&1 | tee "$LOCAL_WORKSPACE/cbm_eval_log_$JOB_ID.txt"

echo "--- SYNCING RESULTS BACK ---"
mkdir -p "$PROJECT_ROOT/results" "$PROJECT_ROOT/logs"
cp "$LOCAL_WORKSPACE/$RESULT_FILE" "$PROJECT_ROOT/$RESULT_FILE"
cp "$LOCAL_WORKSPACE/cbm_eval_log_$JOB_ID.txt" \
   "$PROJECT_ROOT/logs/cbm_eval_${DATASET}_$JOB_ID.txt"

echo "Done. Results at $PROJECT_ROOT/$RESULT_FILE"
