#!/bin/bash -l
#$ -N CBM_Leakage_Run
#$ -o ~/Scratch/xai-concept-leakage/logs/test_run.out
#$ -e ~/Scratch/xai-concept-leakage/logs/test_run.err
#$ -pe smp 4
#$ -l h_rt=42:00:00
#$ -l mem_free=42G
#$ -wd /home/ucakais/Scratch/xai-concept-leakage
#$ -l gpu=1

# --- 1. LOAD MODULES ---
module purge
module unload compilers mpi gcc-libs

module load python3/3.9-gnu-10.2.0
module load gcc-libs/10.2.0

#pytorch
module load cuda/11.8.0/gnu-10.2.0
module load cudnn/9.2.0.82/cuda-11
module load pytorch/2.1.0/gpu

export CC=$(which gcc)
export CXX=$(which g++)


conda activate xai_test

# --- 2. SET THREADING ---
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${NSLOTS}"
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS

# --- 3. PATHS ---
PROJECT_ROOT=~/Scratch/xai-concept-leakage
RESULTS_DIR=$PROJECT_ROOT/results
CONFIG_PATH=$PROJECT_ROOT/experiments/configs/tabulartoy.yaml


# --- 5. RUN PYTHON WORKER ---
python3 -u $PROJECT_ROOT/experiments/run_experiments.py \
    --config $CONFIG_PATH \
    --project_name "Myriad Test Run" \
    --output-dir $RESULTS_DIR

