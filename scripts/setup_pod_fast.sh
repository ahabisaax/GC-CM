#!/bin/bash
# Fast per-pod bootstrap for a parallel sweep shard.
#
# Assumes: repo already cloned at /workspace/GC-CM, celeba.tar already on the
# pod at /workspace/data/celeba.tar (upload it first — that is the slow part).
#
# Puts the venv and the extracted images on LOCAL disk (/root), because
# /workspace is network-backed and ~156x slower on small-file access — and
# CelebA is 202k small JPEGs read every epoch. Results/logs stay on
# /workspace so they survive a pod stop.
#
#   Usage: bash scripts/setup_pod_fast.sh
#   Env:   WANDB_KEY (optional) — written to /workspace/.sweep_env
set -euo pipefail

REPO=/workspace/GC-CM
VENV=/root/venv
DATA=/root/data

echo "== fast pod setup =="
mkdir -p "$DATA" /workspace/results /workspace/logs

# --- venv on local disk
if [ ! -x "$VENV/bin/python" ]; then
    echo "-- creating venv"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
fi
if ! "$VENV/bin/python" -c "import pytorch_lightning" 2>/dev/null; then
    echo "-- installing requirements (torch 2.3.1 / PL 1.9.5)"
    "$VENV/bin/pip" install -q -r "$REPO/requirements.txt"
    "$VENV/bin/pip" install -q -e "$REPO" --no-deps
fi

# --- CelebA onto local disk
if [ ! -d "$DATA/celeba/img_align_celeba" ] || \
   [ "$(ls "$DATA/celeba/img_align_celeba" 2>/dev/null | wc -l)" -lt 200000 ]; then
    echo "-- extracting CelebA to local disk"
    [ -f "$DATA/celeba.tar" ] || cp /workspace/data/celeba.tar "$DATA/celeba.tar"
    tar -xf "$DATA/celeba.tar" -C "$DATA/" 2>/dev/null
fi
echo "-- celeba images: $(ls "$DATA/celeba/img_align_celeba" | wc -l)"

# --- symlink repo data/ -> local payloads
mkdir -p "$REPO/data"
[ -e "$REPO/data/celeba" ] || ln -s "$DATA/celeba" "$REPO/data/celeba"

# --- wandb creds (persist on /workspace so they survive a stop)
if [ -n "${WANDB_KEY:-}" ]; then
    umask 077
    echo "export WANDB_API_KEY=$WANDB_KEY" > /workspace/.sweep_env
    "$VENV/bin/wandb" login "$WANDB_KEY" >/dev/null 2>&1 || true
fi

# --- runpodctl, so AUTOSTOP can stop this pod when its shard finishes
command -v runpodctl >/dev/null 2>&1 || \
    curl -sSL https://cli.runpod.net | bash >/dev/null 2>&1 || true
if [ -f /workspace/.sweep_env ] && command -v runpodctl >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /workspace/.sweep_env
    [ -n "${RUNPOD_API_KEY:-}" ] && runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1 || true
fi

echo "== ready =="
"$VENV/bin/python" -c "import torch,pytorch_lightning as pl;print(f'torch {torch.__version__} PL {pl.__version__} cuda={torch.cuda.is_available()}')"
