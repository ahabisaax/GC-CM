#!/bin/bash -l
#$ -N CBM_Leakage_Run
#$ -o ~/xai-concept-leakage/logs/job_$JOB_ID_$TASK_ID.out
#$ -e ~/xai-concept-leakage/logs/job_$JOB_ID_$TASK_ID.err
#$ -pe smp 4
#$ -l h_rt=42:00:00
#$ -l mem_free=42G
#$ -wd /home/ucakais/xai-concept-leakage
#$ -l gpu=1
#$ -acallow=ucakais

# --- 1. LOAD MODULES ---
module purge
module unload compilers mpi gcc-libs

module load python3/3.11
module load gcc-libs/10.2.0

#pytorch
module load cuda/11.3.1/gnu-10.2.0
module load cudnn/8.2.1.32/cuda-11.3
module load pytorch/1.11.0/gpu

export CC=$(which gcc)
export CXX=$(which g++)


conda activate xai_test

# --- 2. SET THREADING ---
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${NSLOTS}"
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS

# --- 3. PATHS ---
PROJECT_ROOT=~/xai-concept-leakage
RESULTS_DIR=$PROJECT_ROOT/results
CONFIG_PATH=$PROJECT_ROOT/experiments/configs/tabulartoy.yaml

echo "Running task with Python: $(which python --version)"

# --- 5. RUN PYTHON WORKER ---
$PYTHON_BIN -u $PROJECT_ROOT/experiments/run_worker.py \
    --config $CONFIG_PATH \
    --output-dir $RESULTS_DIR