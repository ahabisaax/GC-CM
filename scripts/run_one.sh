#!/bin/bash
# Launch a single training + evaluation run.
#
#   Usage: run_one.sh {model} {dataset} {lambda_c} {seed}
#     model     cem | gc_cem
#     dataset   celeba (fully supported) | any dataset with CONFIG= override
#     lambda_c  0.1 | 0.5 | 1.0
#     seed      0..4   (maps to the pipeline's split/fold index)
#
# Config resolution: for celeba this uses experiments/configs/
# celeba_redone_{cem|gccem}_lam{0_1|0_5|1_0}.yaml. For any other dataset pass
# an explicit config: CONFIG=experiments/configs/foo.yaml run_one.sh ...
#
# Results land where the pipeline already writes them:
#   $RESULTS_DIR/$dataset/<run_name>_split_{seed}_results.joblib
# plus a sentinel $RESULTS_DIR/$dataset/.done_{model}_lam{lambda_c}_seed{seed}
# written only on success — run_sweep.sh uses it to skip completed runs.
#
# Env: WANDB_API_KEY read from the environment (falls back to offline mode),
#      WANDB_PROJECT optionally overrides the config's project_name,
#      DATA_DIR / RESULTS_DIR / LOGS_DIR as in setup_pod.sh.
set -euo pipefail

if [ $# -ne 4 ]; then
    echo "usage: $0 {cem|gc_cem} {dataset} {0.1|0.5|1.0} {seed 0-4}" >&2
    exit 2
fi
MODEL="$1"; DATASET="$2"; LAM="$3"; SEED="$4"

WORKSPACE="${WORKSPACE:-/workspace}"
RESULTS_DIR="${RESULTS_DIR:-$WORKSPACE/results}"
LOGS_DIR="${LOGS_DIR:-$WORKSPACE/logs}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# Activate the venv if the caller has not — otherwise a bare `python` silently
# resolves to system Python and fails on the first missing import.
if [ -z "${VIRTUAL_ENV:-}" ]; then
    for _v in "${VENV_DIR:-}" /root/venv "$WORKSPACE/venv"; do
        if [ -n "$_v" ] && [ -f "$_v/bin/activate" ]; then
            # shellcheck disable=SC1091
            source "$_v/bin/activate"; break
        fi
    done
fi

# ------------------------------------------------------------ config lookup
if [ -z "${CONFIG:-}" ]; then
    case "$DATASET" in
        celeba)
            case "$MODEL" in
                cem)    model_tag="cem" ;;
                gc_cem) model_tag="gccem" ;;
                *) echo "unknown model '$MODEL' (want cem|gc_cem)" >&2; exit 2 ;;
            esac
            case "$LAM" in
                0.1) lam_tag="lam0_1" ;;
                0.5) lam_tag="lam0_5" ;;
                1.0|1) lam_tag="lam1_0" ;;
                *) echo "unknown lambda_c '$LAM' (want 0.1|0.5|1.0)" >&2; exit 2 ;;
            esac
            CONFIG="experiments/configs/celeba_redone_${model_tag}_${lam_tag}.yaml"
            ;;
        *)
            echo "No config convention for dataset '$DATASET'." >&2
            echo "Pass one explicitly: CONFIG=experiments/configs/<file>.yaml $0 ..." >&2
            exit 2
            ;;
    esac
fi
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 2; }

# ------------------------------------------------------------- resumability
OUT_DIR="$RESULTS_DIR/$DATASET"
SENTINEL="$OUT_DIR/.done_${MODEL}_lam${LAM}_seed${SEED}"
if [ -f "$SENTINEL" ]; then
    echo "SKIP $MODEL $DATASET lam=$LAM seed=$SEED (sentinel exists: $SENTINEL)"
    exit 0
fi
mkdir -p "$OUT_DIR" "$LOGS_DIR"

# ------------------------------------------------------------------- wandb
# Non-interactive shells (nohup/ssh) don't read .bashrc, so pick the key up
# from the persistent env file. Note ~/.netrc lives on container disk and is
# lost on pod stop, whereas $WORKSPACE persists — so the env file is the
# durable source of truth.
for _envf in "${SWEEP_ENV:-}" "$WORKSPACE/.sweep_env" /workspace/.sweep_env; do
    if [ -n "$_envf" ] && [ -f "$_envf" ]; then
        # shellcheck disable=SC1090
        source "$_envf"; break
    fi
done
# netrc is equally valid auth for wandb — only go offline if neither exists.
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    echo "WARNING: no W&B credentials (env or netrc) — running offline" >&2
    export WANDB_MODE=offline
else
    export WANDB_MODE="${WANDB_MODE:-online}"
fi
PROJECT_ARGS=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    PROJECT_ARGS=(--project_name "$WANDB_PROJECT")
fi

# --------------------------------------------------------------------- run
LOG_FILE="$LOGS_DIR/${DATASET}_${MODEL}_lam${LAM}_seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
echo "RUN  $MODEL $DATASET lam=$LAM seed=$SEED"
echo "     config=$CONFIG  out=$OUT_DIR  log=$LOG_FILE"

# seed maps to the split index: run exactly split $SEED.
# checkpoint_monitor/mode: apply the same checkpoint-selection criterion
# (validation concept accuracy) to every model in a sweep.
python -u experiments/run_experiments.py \
    --config "$CONFIG" \
    --output_dir "$OUT_DIR" \
    "${PROJECT_ARGS[@]}" \
    -p start_split "$SEED" \
    -p trials "$((SEED + 1))" \
    -p checkpoint_monitor val_c_accuracy \
    -p checkpoint_mode max \
    2>&1 | tee "$LOG_FILE"

touch "$SENTINEL"
echo "DONE $MODEL $DATASET lam=$LAM seed=$SEED"
