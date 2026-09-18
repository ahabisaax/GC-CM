#!/bin/bash
# Bootstrap a RunPod instance for CUB-200 runs.
#
#   Usage: bash scripts/setup_pod_cub.sh
#   Env:   WANDB_KEY, REPO (default /workspace/GC-CM), DATA (default /root/data)
#
# CUB is not in the repo (data/ is gitignored) and has no download script, but
# the raw images ARE fetchable: Caltech DATA serves CUB_200_2011.tgz via a
# redirect to signed S3, which wget follows fine.
#
# The class_attr_data_10 pickles (Koh et al. splits, 48MB) are NOT in the raw
# archive and must be uploaded separately with scp before running this.
#
# cub_loader.py:836 rewrites the authors' absolute paths by string-replacing
#   /juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets/  ->  root_dir
# and the configs set root_dir: data/CUB200/, resolved from the repo root. So
# the required layout is data/CUB200/{CUB_200_2011,class_attr_data_10}.
#
# Heavy data goes on /root (local disk): /workspace is MooseFS and CUB is
# 11,788 small JPEGs read every epoch. Only the two data dirs are symlinked,
# so cub_loader.py stays in the repo where `import data.CUB200.cub_loader`
# expects it.
set -euo pipefail

REPO="${REPO:-/workspace/GC-CM}"
VENV="${VENV:-/root/venv}"
DATA="${DATA:-/root/data}"
CUB="$DATA/CUB200"
URL="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
N_EXPECTED=11788

echo "== CUB pod setup =="
mkdir -p "$CUB" /workspace/results /workspace/logs

missing=""
for t in wget tar rsync tmux; do command -v "$t" >/dev/null 2>&1 || missing="$missing $t"; done
if [ -n "$missing" ]; then
    echo "-- installing:$missing"
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

if [ ! -d "$CUB/CUB_200_2011/images" ]; then
    [ -f "$CUB/CUB_200_2011.tgz" ] || { echo "-- downloading CUB (~1.1GB)"; wget -c --progress=dot:giga -O "$CUB/CUB_200_2011.tgz" "$URL"; }
    echo "-- extracting"; tar -xzf "$CUB/CUB_200_2011.tgz" -C "$CUB/"
fi

n_img=$(find "$CUB/CUB_200_2011/images" -name '*.jpg' | wc -l | tr -d ' ')
echo "-- images: $n_img (expect $N_EXPECTED)"
[ "$n_img" -eq "$N_EXPECTED" ] || echo "   WARNING: image count mismatch" >&2

if [ ! -f "$CUB/class_attr_data_10/train.pkl" ]; then
    echo "   NOTE: class_attr_data_10/*.pkl not present yet - scp them to $CUB/class_attr_data_10/" >&2
fi

mkdir -p "$REPO/data/CUB200"
[ -e "$REPO/data/CUB200/CUB_200_2011" ]     || ln -s "$CUB/CUB_200_2011"     "$REPO/data/CUB200/CUB_200_2011"
[ -e "$REPO/data/CUB200/class_attr_data_10" ] || ln -s "$CUB/class_attr_data_10" "$REPO/data/CUB200/class_attr_data_10"

if [ -n "${WANDB_KEY:-}" ]; then
    umask 077; echo "export WANDB_API_KEY=$WANDB_KEY" > /workspace/.sweep_env
    "$VENV/bin/wandb" login "$WANDB_KEY" >/dev/null 2>&1 || true
fi

echo "== ready =="
"$VENV/bin/python" -c "import torch,pytorch_lightning as pl;print(f'torch {torch.__version__} PL {pl.__version__} cuda={torch.cuda.is_available()}')"
