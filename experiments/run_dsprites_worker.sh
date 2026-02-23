#!/bin/bash -l
#$ -N CBM_Leakage_Run
#$ -o ~/Scratch/xai-crcbm/logs/dsprites_$JOB_ID.out
#$ -e ~/Scratch/xai-crcbm/logs/dsprites_$JOB_ID.err
#$ -pe smp 8
#$ -l h_rt=10:00:00
#$ -l mem=8G
#$ -l tmpfs=20G
#$ -wd /home/ucakais/Scratch/xai-crcbm
#$ -l gpu=1

# --- 1. LOAD MODULES ---
module purge
module unload compilers mpi gcc-libs

module load python3/3.9-gnu-10.2.0
module load gcc-libs/10.2.0

export CC=$(which gcc)
export CXX=$(which g++)

echo "Sourcing conda setup..."
conda activate xai2

# --- 2. SET THREADING ---
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${NSLOTS}"
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS

# --- 3. PATHS ---
PROJECT_ROOT=~/Scratch/xai-crcbm
#CHANGE FOR EACH DATASET
RESULTS_DIR=$PROJECT_ROOT/results/dsprites_dep_0_models_crcem_lr_checks
CONFIG_PATH=$PROJECT_ROOT/experiments/configs/dsprites.yaml

export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
# --- 5. RUN PYTHON WORKER ---
$CONDA_PREFIX/bin/python -u $PROJECT_ROOT/experiments/run_experiments.py \
    --config $CONFIG_PATH \
    --project_name "Dsprites" \
    --output_dir $RESULTS_DIR
