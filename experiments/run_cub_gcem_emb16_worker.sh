#!/bin/bash -l
#$ -N CUB_GCEM_emb16
#$ -o ~/Scratch/xai-crcbm/logs/CUB_GCEM_emb16_$JOB_ID.out
#$ -e ~/Scratch/xai-crcbm/logs/CUB_GCEM_emb16_$JOB_ID.err
#$ -pe smp 8
#$ -l h_rt=24:00:00
#$ -l mem=24G
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

echo "Sourcing conda setup..."
conda activate xai2

# --- 2. THREADING ---
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${NSLOTS}"
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS
export NUMEXPR_NUM_THREADS=$NSLOTS

# --- 3. PATHS ---
PROJECT_ROOT=~/Scratch/xai-crcbm
TRAIN_RESULTS_DIR=$PROJECT_ROOT/results/cub_gcem_emb16
DATASET_TAR="CUB200.tar"
CEM_CKPT_PATH=$PROJECT_ROOT/results/cub_cem/CEM_adam_lr1e-03_bs256_lam_c0.1/CEM_adam_lr1e-03_bs256_lam_c0.1_fold_1.pt

echo "--- SETTING UP FAST LOCAL WORKSPACE ---"
LOCAL_WORKSPACE="$TMPDIR/$JOB_ID"
mkdir -p "$LOCAL_WORKSPACE/data"
mkdir -p "$LOCAL_WORKSPACE/results"

# Copy code (exclude large dirs)
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

# Copy existing CEM fold_1 checkpoint so we can extract its embeddings too
echo "Copying CEM fold_1 checkpoint..."
CEM_LOCAL_DIR="$LOCAL_WORKSPACE/results/cub_cem/CEM_adam_lr1e-03_bs256_lam_c0.1"
mkdir -p "$CEM_LOCAL_DIR"
if [ -f "$CEM_CKPT_PATH" ]; then
    cp "$CEM_CKPT_PATH" "$CEM_LOCAL_DIR/"
else
    echo "WARNING: CEM checkpoint not found at $CEM_CKPT_PATH — will skip CEM in CVL eval"
fi

# Copy existing CVL results cache to enable incremental computation
if [ -f "$PROJECT_ROOT/results/results_cvl_gcem_emb16.dict" ]; then
    cp "$PROJECT_ROOT/results/results_cvl_gcem_emb16.dict" "$LOCAL_WORKSPACE/results/"
fi

cd "$LOCAL_WORKSPACE"
export PYTHONPATH="$LOCAL_WORKSPACE:$PYTHONPATH"
echo "Running from: $(pwd)"
echo "NSLOTS=$NSLOTS  OMP_NUM_THREADS=$OMP_NUM_THREADS"

# --- 4. TRAIN GC-CEM (emb16, lam_c=0.1, 1 fold) ---
echo ""
echo "========================================================"
echo "STEP 1: Training GC-CEM (emb_size=16, lam_c=0.1, 1 fold)"
echo "========================================================"
LOCAL_TRAIN_RESULTS="$LOCAL_WORKSPACE/results/cub_gcem_emb16"

$CONDA_PREFIX/bin/python -u experiments/run_experiments.py \
    --config experiments/configs/cub_gcem_emb16_lam01.yaml \
    --project_name "CUB_GCEM_emb16" \
    --output_dir "$LOCAL_TRAIN_RESULTS"

echo "Training complete."

# --- 5. FIND TRAINED CHECKPOINT ---
echo ""
echo "========================================================"
echo "STEP 2: Locating trained checkpoint"
echo "========================================================"
GCEM_CKPT=$(find "$LOCAL_TRAIN_RESULTS" -name "*.pt" | sort | head -1)
if [ -z "$GCEM_CKPT" ]; then
    echo "ERROR: No .pt checkpoint found in $LOCAL_TRAIN_RESULTS"
    echo "Contents of $LOCAL_TRAIN_RESULTS:"
    find "$LOCAL_TRAIN_RESULTS" -type f
    exit 1
fi
echo "GC-CEM checkpoint: $GCEM_CKPT"

# Determine CEM checkpoint (local copy)
CEM_LOCAL_CKPT="$CEM_LOCAL_DIR/CEM_adam_lr1e-03_bs256_lam_c0.1_fold_1.pt"
if [ -f "$CEM_LOCAL_CKPT" ]; then
    CEM_ARG="--cem_ckpt $CEM_LOCAL_CKPT"
    echo "CEM checkpoint:    $CEM_LOCAL_CKPT"
else
    CEM_ARG=""
    echo "CEM checkpoint not available — GC-CEM only"
fi

# --- 6. EXTRACT EMBEDDINGS (dataset still on local SSD) ---
echo ""
echo "========================================================"
echo "STEP 3: Extracting embeddings (val+test clean split)"
echo "========================================================"
EMBED_CACHE="$LOCAL_WORKSPACE/results/cub_embeddings_gcem_emb16.joblib"

$CONDA_PREFIX/bin/python -u experiments/evaluate_models/cache_cub_embeddings_single.py \
    --gcem_ckpt "$GCEM_CKPT" \
    $CEM_ARG \
    --output    "$EMBED_CACHE" \
    --data_root "$LOCAL_WORKSPACE/data/CUB200/"

if [ $? -ne 0 ]; then
    echo "ERROR: Embedding extraction failed"
    exit 1
fi

# --- 7. COMPUTE CVL / ICVL VARIANTS ---
echo ""
echo "========================================================"
echo "STEP 4: Computing CVL/ICVL variants"
echo "========================================================"
CVL_RESULTS="$LOCAL_WORKSPACE/results/results_cvl_gcem_emb16.dict"

$CONDA_PREFIX/bin/python -u experiments/evaluate_models/cvl_variants_cub.py \
    --embed_cache "$EMBED_CACHE" \
    --save_path   "$CVL_RESULTS"

if [ $? -ne 0 ]; then
    echo "ERROR: CVL/ICVL evaluation failed"
    exit 1
fi

# --- 8. SYNC ALL RESULTS BACK TO SCRATCH ---
echo ""
echo "========================================================"
echo "STEP 5: Syncing results back to Scratch"
echo "========================================================"
mkdir -p "$TRAIN_RESULTS_DIR"
mkdir -p "$PROJECT_ROOT/results"

# Training checkpoints and split results
rsync -a "$LOCAL_TRAIN_RESULTS/" "$TRAIN_RESULTS_DIR/"

# Embedding cache
cp "$EMBED_CACHE" "$PROJECT_ROOT/results/cub_embeddings_gcem_emb16.joblib"

# CVL/ICVL results dict
cp "$CVL_RESULTS" "$PROJECT_ROOT/results/results_cvl_gcem_emb16.dict"

echo ""
echo "Done."
echo "  Training results → $TRAIN_RESULTS_DIR"
echo "  Embeddings       → $PROJECT_ROOT/results/cub_embeddings_gcem_emb16.joblib"
echo "  CVL/ICVL results → $PROJECT_ROOT/results/results_cvl_gcem_emb16.dict"
