#!/bin/bash
# Bootstrap a RunPod instance for the AwA2 5-fold runs.
#
#   Usage: bash scripts/setup_pod_awa2.sh
#   Env:   WANDB_KEY   (optional) written to /workspace/.sweep_env
#          REPO        default /workspace/GC-CM
#          DATA        default /root/data      (LOCAL disk, see below)
#
# Venv and dataset go on LOCAL disk (/root). /workspace is network-backed
# (MooseFS): measured 156x slower on small-file creation and 82x slower on
# AwA2 image reads, and AwA2 training is dataloader-bound either way.
# Results and logs stay on /workspace so they survive a pod stop.
#
# Idempotent: re-running skips the venv, the download and the extraction if
# they are already in place.
set -euo pipefail

REPO="${REPO:-/workspace/GC-CM}"
VENV="${VENV:-/root/venv}"
DATA="${DATA:-/root/data}"
AWA="$DATA/AwA2"
URL="https://cvml.ista.ac.at/AwA2/AwA2-data.zip"
N_EXPECTED=37322

echo "== AwA2 pod setup =="
mkdir -p "$AWA" /workspace/results /workspace/logs

# The RunPod pytorch images ship without unzip (and often without tmux), and
# set -e turns a missing one into a silent-looking death 13GB into the job.
missing=""
for t in unzip wget rsync tmux; do command -v "$t" >/dev/null 2>&1 || missing="$missing $t"; done
if [ -n "$missing" ]; then
    echo "-- installing:$missing"
    apt-get update -qq >/dev/null 2>&1 || true
    # shellcheck disable=SC2086
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $missing >/dev/null 2>&1 || true
fi
for t in unzip wget; do
    command -v "$t" >/dev/null 2>&1 || { echo "FATAL: $t unavailable and could not be installed" >&2; exit 1; }
done

# --- venv on local disk (torch 2.3.1 / PL 1.9.5: the GC models use the
#     optimizer_idx multi-optimizer API removed in Lightning 2.0)
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

# --- AwA2 straight onto local disk (~13GB archive, ~14GB extracted)
if [ ! -d "$AWA/Animals_with_Attributes2/JPEGImages" ]; then
    if [ ! -f "$AWA/AwA2-data.zip" ]; then
        echo "-- downloading AwA2 (~13GB)"
        wget -c --progress=dot:giga -O "$AWA/AwA2-data.zip" "$URL"
    else
        echo "-- archive present, resuming any partial download"
        wget -c --progress=dot:giga -O "$AWA/AwA2-data.zip" "$URL" || true
    fi
    echo "-- extracting"
    unzip -q -o "$AWA/AwA2-data.zip" -d "$AWA/"
fi

BASE="$AWA/Animals_with_Attributes2"
for f in classes.txt predicates.txt predicate-matrix-binary.txt; do
    [ -f "$BASE/$f" ] || { echo "MISSING $BASE/$f" >&2; exit 1; }
done
n_img=$(find "$BASE/JPEGImages" -name '*.jpg' | wc -l | tr -d ' ')
echo "-- images: $n_img (expect $N_EXPECTED)"
[ "$n_img" -eq "$N_EXPECTED" ] || echo "   WARNING: image count mismatch (partial extract?)" >&2

# A stale bad-image cache from an interrupted extract would permanently
# exclude images that are actually fine. Drop it; the loader rebuilds it.
rm -f "$BASE/.awa2_bad_images.txt"

# --- point the repo's data/ at the local copy
mkdir -p "$REPO/data"
[ -e "$REPO/data/AwA2" ] || ln -s "$AWA" "$REPO/data/AwA2"

# --- wandb creds (on /workspace so they survive a stop; ~/.netrc does not)
if [ -n "${WANDB_KEY:-}" ]; then
    umask 077
    echo "export WANDB_API_KEY=$WANDB_KEY" > /workspace/.sweep_env
    "$VENV/bin/wandb" login "$WANDB_KEY" >/dev/null 2>&1 || true
fi

echo "== ready =="
nproc_q=$(awk '/^cpu.max/ {print}' /sys/fs/cgroup/cpu.max 2>/dev/null || true)
echo "-- nproc reports $(nproc); cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo unknown)"
echo "   (nproc shows HOST cores, not this container's quota - set NUM_WORKERS from the quota)"
"$VENV/bin/python" -c "import torch,pytorch_lightning as pl;print(f'torch {torch.__version__} PL {pl.__version__} cuda={torch.cuda.is_available()}')"
