#!/bin/bash -l
# --- 1. SGE JOB CONFIGURATION ---
#
# (Job Name)
#$ -N CBM_Leakage_Run
#
# (Standard Output log - place in a 'logs' folder)
#$ -o ~/xai-concept-leakage/logs/job_$JOB_ID_$TASK_ID.out
#
# (Standard Error log)
#$ -e ~/xai-concept-leakage/logs/job_$JOB_ID_$TASK_ID.err
#
# (Request 4 CPU cores)
#$ -pe smp 4
#
# (Max runtime - e.g., 42 hours)
#$ -l h_rt=42:00:00
#
# (Request 42GB total memory)
#$ -l mem_free=42G
#
# (Working directory - IMPORTANT: REPLACE 'ucakmc' with your UCL username)
#$ -wd /home/ucakais/xai-concept-leakage
#
# (Request GPU - Myriad)
#$ -l gpu=1
#$ -acallow=ucakais  # <--- !! IMPORTANT: SET YOUR UCL ACCOUNT !!

# --- 2. LOAD MODULES & ENVIRONMENT ---
echo "Loading modules..."
module purge
module load python/miniconda3/4.10.3
module load nvidia/525.125.06 # Or the appropriate CUDA/NVIDIA driver

echo "Activating conda env..."
conda activate xai_env

export CC=$(which gcc)
export CXX=$(which g++)



# --- 3. SET PARALLELISM ---
# $NSLOTS is the SGE variable for the number of cores requested
echo "Setting parallel flags for $NSLOTS cores..."
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${NSLOTS}"
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS

# --- 4. DEFINE PATHS ---
PROJECT_ROOT=~/xai-concept-leakage
# Top-level results dir - a new folder will be made inside this
RESULTS_DIR=$PROJECT_ROOT/results/
CONFIG_PATH=$1 # Get the config path passed from qsub

# Get the SGE task ID
TASK_ID=$SGE_TASK_ID
echo "Running SGE Task ID: $TASK_ID"

# --- 5. RUN THE PYTHON WORKER SCRIPT ---
# This is the new Python script that can run a *single* task
echo "Starting Python worker script..."
python -u $PROJECT_ROOT/experiments/run_worker.py \
    --config $CONFIG_PATH \
    --task-id $TASK_ID \
    --output-dir $RESULTS_DIR \
    # --force-cpu # Removed this, as we are requesting a GPU

echo "Task $TASK_ID complete."