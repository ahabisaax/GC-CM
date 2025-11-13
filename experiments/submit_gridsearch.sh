#!/bin/bash -l

# --- 1. CONFIGURATION ---
PROJECT_ROOT=~/xai-concept-leakage
EXPERIMENT_CONFIG=$PROJECT_ROOT/experiments/configs/tabulartoy.yaml

# --- 2. COUNT TASKS ---
NUM_TASKS=$(python3 - <<EOF
import yaml, copy, sys
try:
    with open("$EXPERIMENT_CONFIG") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
except FileNotFoundError:
    print(f"Error: Config file not found at {EXPERIMENT_CONFIG}", file=sys.stderr)
    sys.exit(1)

shared_params = config.get("shared_params", {})
if "runs" not in config:
    print(f"Error: 'runs' key not found", file=sys.stderr)
    sys.exit(1)

total_runs = 0
for current_config in config["runs"]:
    run_config = copy.deepcopy(shared_params)
    run_config.update(current_config)
    list_params = {k: v for k, v in run_config.items() if isinstance(v, list)}
    if not list_params:
        total_runs += 1
    else:
        perms = 1
        for v in list_params.values():
            perms *= len(v)
        total_runs += perms

print(total_runs)
EOF
)

if [ $? -ne 0 ] || [ -z "$NUM_TASKS" ] || [ "$NUM_TASKS" -eq 0 ]; then
    echo "Error: Failed to count tasks. Check your YAML."
    exit 1
fi

echo "Found $NUM_TASKS total experiments in $EXPERIMENT_CONFIG"
echo "Submitting SGE job array..."

# --- 3. SUBMIT THE ARRAY JOB ---
qsub -t 1-$NUM_TASKS $PROJECT_ROOT/experiments/run_worker.sh $EXPERIMENT_CONFIG

echo "Job array submitted."
