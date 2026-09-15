#!/bin/bash -l
#$ -N AwA2_CBM_HardSeq
#$ -o /home/ucakais/Scratch/xai-crcbm/logs/AwA2_CBM_HardSeq_$JOB_ID.out
#$ -e /home/ucakais/Scratch/xai-crcbm/logs/AwA2_CBM_HardSeq_$JOB_ID.err
#$ -pe smp 8
#$ -l h_rt=24:00:00
#$ -l mem=4G
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
# WALLTIME WARNING: 24h is likely not enough for both. Measured elsewhere,
# a 120-epoch CBM fold is ~2.1h on an A40 and HPC folds have run slower;
# Sequential trains two stages (x2c then c2y, 120 epochs each) so its folds
# are the longest in the set. Hard CBM runs FIRST so it completes. If the
# job is killed at walltime partway through Sequential, just resubmit:
# run_experiments loads any split whose results joblib already exists and
# carries on from the first missing fold.
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

run_cfg hardcbm experiments/configs/awa2_hardcbm_5fold.yaml
run_cfg seqcbm  experiments/configs/awa2_seqcbm_5fold.yaml

echo "=== syncing wandb offline runs ==="
mkdir -p "$PROJECT_ROOT/wandb"
[ -d "$LOCAL_WORKSPACE/wandb" ] && rsync -a "$LOCAL_WORKSPACE/wandb/" "$PROJECT_ROOT/wandb/" || true
echo "=== $(date -u +%FT%TZ) WORKER COMPLETE ==="
