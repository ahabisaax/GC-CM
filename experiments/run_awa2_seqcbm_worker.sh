#!/bin/bash -l
#$ -N AwA2_SeqCBM_5fold
#$ -o /home/ucakais/Scratch/xai-crcbm/logs/AwA2_SeqCBM_$JOB_ID.out
#$ -e /home/ucakais/Scratch/xai-crcbm/logs/AwA2_SeqCBM_$JOB_ID.err
#$ -pe smp 8
#$ -l h_rt=16:00:00
#$ -l mem=6G
#$ -l tmpfs=40G
#$ -wd /home/ucakais/Scratch/xai-crcbm
#$ -l gpu=1
#$ -P Gold
#$ -A hpc.28
#
# Hard CBM then Sequential CBM, 5 folds each, lr 1e-3, 120 epochs.
# Results land in results/awa2_5fold_cbm, the same folder name the RunPod
# CBM runs write to, so the two merge cleanly.
#
# Sequential CBM only, 5 folds, 100 epochs, lr 1e-3.
#
# Cost per fold is one x2c training run plus two overheads, NOT two training
# runs: the c2y stage trains a linear head on a TensorDataset of precomputed
# concept predictions (training.py:943), with no backbone in the loop. The
# overheads are the concept-precompute pass, which runs at batch_size=1 over
# ~26k images (training.py:930), and sequential materialising the whole
# training set in RAM (~5.1GB for AwA2 at 128px, training.py:780) - hence
# mem=6G per slot.
#
# 16h for ~2.5h/fold is workable but not generous. Results are written
# straight to Scratch, so a walltime kill costs the fold in flight, not the
# ones already done; resubmitting the identical script resumes from the first
# missing fold because run_experiments loads any split whose results joblib
# already exists.
#
# Intervention curves now reach W&B for sequential runs (9243d5d): the
# sequential path records its W&B run id, which run_experiments needs to
# resume the run and attach the curves. Earlier sequential runs computed the
# curves but never logged them.
#
# That resume only works because results are written straight to Scratch
# rather than staged in $TMPDIR and rsynced at the end - a walltime kill
# would take unsynced folds with it. The dataset is still staged locally,
# which is the part that actually matters for throughput.

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

PROJECT_ROOT=/home/ucakais/Scratch/xai-crcbm
FINAL_RESULTS_DIR=$PROJECT_ROOT/results/awa2_5fold_cbm
# The older worker referenced $LOCAL_WORKSPACE without ever setting it.
# Define it from the scheduler's $TMPDIR (node-local disk) explicitly.
LOCAL_WORKSPACE="${TMPDIR:?TMPDIR not set - are we running under qsub?}"

mkdir -p "$FINAL_RESULTS_DIR" "$PROJECT_ROOT/logs" "$LOCAL_WORKSPACE/data"

echo "=== staging AwA2 to $LOCAL_WORKSPACE/data (node-local disk) ==="
cp -r "$PROJECT_ROOT/data/AwA2" "$LOCAL_WORKSPACE/data/AwA2"
n_img=$(find "$LOCAL_WORKSPACE/data/AwA2" -name '*.jpg' | wc -l)
echo "  staged: $n_img images (expect 37322)"
# A stale cache from an interrupted copy would permanently drop good images.
rm -f "$LOCAL_WORKSPACE/data/AwA2/Animals_with_Attributes2/.awa2_bad_images.txt"

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export WANDB_MODE=online

run_cfg () {
    tag="$1"; cfg="$2"
    if [ ! -f "$cfg" ]; then echo "SKIP $tag: missing $cfg"; return; fi
    echo "=== $(date -u +%FT%TZ) START $tag ($cfg) ==="
    $CONDA_PREFIX/bin/python -u experiments/run_experiments.py \
        --config "$cfg" \
        --project_name "AwA2" \
        --output_dir "$FINAL_RESULTS_DIR" \
        -p dataset_config.root_dir "$LOCAL_WORKSPACE/data/" \
        -p dataset_config.num_workers "$NSLOTS"
    rc=$?
    echo "=== $(date -u +%FT%TZ) FINISH $tag rc=$rc ==="
}

run_cfg seqcbm  experiments/configs/awa2_seqcbm_5fold.yaml

echo "=== syncing wandb offline runs ==="
mkdir -p "$PROJECT_ROOT/wandb"
[ -d "$LOCAL_WORKSPACE/wandb" ] && rsync -a "$LOCAL_WORKSPACE/wandb/" "$PROJECT_ROOT/wandb/" || true
echo "=== $(date -u +%FT%TZ) WORKER COMPLETE ==="
