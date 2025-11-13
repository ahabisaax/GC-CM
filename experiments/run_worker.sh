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
module load pytorch/2.1.0/gpu
module load gcc-libs/10.2.0

# --- 2. SET THREADING ---
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${NSLOTS}"
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS

# --- 3. PATHS ---
PROJECT_ROOT=~/xai-concept-leakage
RESULTS_DIR=$PROJECT_ROOT/results
CONFIG_PATH=$1
TASK_ID=$SGE_TASK_ID

# --- 4. PYTHON BINARY ---
PYTHON_BIN=/shared/ucl/apps/python/3.9.6/gnu-10.2.0/bin/python3.9

echo "Running task $TASK_ID with Python: $($PYTHON_BIN --version)"

# --- 5. RUN PYTHON WORKER ---
$PYTHON_BIN -u $PROJECT_ROOT/experiments/run_worker.py \
    --config $CONFIG_PATH \
    --task-id $TASK_ID \
    --output-dir $RESULTS_DIR

echo "Task $TASK_ID complete."
