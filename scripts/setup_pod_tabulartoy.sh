#!/bin/bash
# Bootstrap a RunPod instance for TabularToy runs.
#
#   Usage: bash scripts/setup_pod_tabulartoy.sh
#   Env:   WANDB_KEY, REPO (default /workspace/GC-CM)
#
# TabularToy is generated, not downloaded, and data/ is gitignored, so the
# dataset (~1.4MB) is scp'd in rather than regenerated: generate_tabulartoy_
# dataset.py would produce a different draw and the runs would no longer sit
# on the same data as everything before them.
#
# Everything lives on /workspace here. The MooseFS penalty that mattered for
# AwA2 and CUB is irrelevant at this size: TabularToy is 9 small CSVs read
# once, not tens of thousands of JPEGs read every epoch.
set -euo pipefail

REPO="${REPO:-/workspace/GC-CM}"
VENV="${VENV:-/root/venv}"

echo "== TabularToy pod setup =="
mkdir -p /workspace/results /workspace/logs

missing=""
for t in rsync tmux; do command -v "$t" >/dev/null 2>&1 || missing="$missing $t"; done
if [ -n "$missing" ]; then
    apt-get update -qq >/dev/null 2>&1 || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $missing >/dev/null 2>&1 || true
fi

if [ ! -x "$VENV/bin/python" ]; then
    echo "-- creating venv"; python3 -m venv "$VENV"; "$VENV/bin/pip" install -q --upgrade pip
fi
if ! "$VENV/bin/python" -c "import pytorch_lightning" 2>/dev/null; then
    echo "-- installing requirements (torch 2.3.1 / PL 1.9.5)"
    "$VENV/bin/pip" install -q -r "$REPO/requirements.txt"
    "$VENV/bin/pip" install -q -e "$REPO" --no-deps
fi

D="$REPO/data/TabularToy/tabulartoy_25_10k"
n=$(ls "$D" 2>/dev/null | wc -l | tr -d ' ')
echo "-- dataset files present: $n (expect 9)"
[ "$n" -eq 9 ] || echo "   NOTE: scp data/TabularToy/tabulartoy_25_10k/ to $D" >&2

if [ -n "${WANDB_KEY:-}" ]; then
    umask 077; echo "export WANDB_API_KEY=$WANDB_KEY" > /workspace/.sweep_env
    "$VENV/bin/wandb" login "$WANDB_KEY" >/dev/null 2>&1 || true
fi

echo "== ready =="
"$VENV/bin/python" -c "import torch,pytorch_lightning as pl;print(f'torch {torch.__version__} PL {pl.__version__} cuda={torch.cuda.is_available()}')"
