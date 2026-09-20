#!/bin/bash -l
#$ -N AwA2_CBM_Fixups
#$ -o /home/ucakais/Scratch/xai-crcbm/logs/AwA2_CBM_Fixups_$JOB_ID.out
#$ -e /home/ucakais/Scratch/xai-crcbm/logs/AwA2_CBM_Fixups_$JOB_ID.err
#$ -pe smp 8
#$ -l h_rt=20:00:00
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
# Two fix-ups to the AwA2 CBM results, in one job:
#   1. Sequential CBM fold 5, missing because the original run hit walltime.
#   2. GC-CBM lam_c 0.1 folds 1-4 re-trained at 120 epochs, so the config
#      stops mixing 90-epoch folds with a 120-epoch fold 5.
#
# Both write into results/awa2_5fold_cbm alongside the existing folds and log
# to the AwA2 W&B project. Sequential now also logs its intervention curves
# (9243d5d); before that only train_end_to_end_model recorded the W&B run id
# that run_experiments needs to attach them.
#
# ~1 Seq fold (100 ep) + 4 GC-CBM folds (120 ep) ~= 14h, so 20h walltime.
# Results go straight to Scratch, so a walltime kill costs only the fold in
# flight and a resubmit resumes from the first missing fold.
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
    tag="$1"; cfg="$2"; shift 2      # remaining args are extra -p overrides
    if [ ! -f "$cfg" ]; then echo "SKIP $tag: missing $cfg"; return; fi
    echo "=== $(date -u +%FT%TZ) START $tag ($cfg) $* ==="
    # Both configs set project_name: AwA2 in shared_params, which is what the
    # existing AwA2 runs used. --project_name does NOT override that (a CLI
    # flag lost to shared_params is how the CUB runs landed in the wrong
    # project), so these log to AwA2 either way.
    $CONDA_PREFIX/bin/python -u experiments/run_experiments.py \
        --config "$cfg" \
        --project_name "AwA2" \
        --output_dir "$FINAL_RESULTS_DIR" \
        -p dataset_config.root_dir "$LOCAL_WORKSPACE/data/" \
        -p dataset_config.num_workers "$NSLOTS" \
        "$@"
    rc=$?
    echo "=== $(date -u +%FT%TZ) FINISH $tag rc=$rc ==="
}

# 1) Sequential CBM fold 5. Folds 1-4 exist; the original run hit its
#    walltime before fold 5. start_split 4 / trials 5 -> split 4 only.
run_cfg seqcbm_fold5 experiments/configs/awa2_seqcbm_5fold.yaml \
        -p start_split 4 -p trials 5

# 2) GC-CBM lam_c 0.1 folds 1-4 at 120 epochs. Those folds were trained at
#    90 while fold 5 (re-run later) got 120, so the 5-fold mean currently
#    mixes budgets inside one config. start_split 0 / trials 4 -> splits
#    0,1,2,3. The config is already at 120 epochs; fold 5 is left alone.
#    --rerun is required: without it run_experiments loads the existing
#    90-epoch split joblibs from cache and trains nothing. It only applies to
#    splits in range, so fold 5 (split 4) is untouched.
run_cfg gccbm_lam0.1_folds1to4 experiments/configs/awa2_gccbm_5fold_lam0_1.yaml \
        --rerun -p start_split 0 -p trials 4

echo "=== syncing wandb offline runs ==="
mkdir -p "$PROJECT_ROOT/wandb"
[ -d "$LOCAL_WORKSPACE/wandb" ] && rsync -a "$LOCAL_WORKSPACE/wandb/" "$PROJECT_ROOT/wandb/" || true
echo "=== $(date -u +%FT%TZ) WORKER COMPLETE ==="
