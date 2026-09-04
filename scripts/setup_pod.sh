#!/bin/bash
# Idempotent RunPod pod setup for the xai-concept-leakage sweeps.
#
# Layout (all overridable via env vars):
#   WORKSPACE    default /workspace         (persistent network volume)
#   DATA_DIR     default $WORKSPACE/data    (datasets, survive pod restarts)
#   RESULTS_DIR  default $WORKSPACE/results
#   LOGS_DIR     default $WORKSPACE/logs
#   VENV_DIR     default $WORKSPACE/venv
#
# Datasets fetched (skipped when the target directory already exists):
#   CUB        -> $DATA_DIR/CUB200/{CUB_200_2011, class_attr_data_10}
#   CelebA     -> $DATA_DIR/celeba   (from $DATA_DIR/celeba.tar if present,
#                 else torchvision download; gdrive quota may block the latter —
#                 in that case upload celeba.tar to the volume and re-run)
#   dSprites   -> $DATA_DIR/dsprites (raw npz + generated dsprites_dep_0.npz;
#                 generation installs tensorflow-cpu on demand)
#   TabularToy -> $DATA_DIR/TabularToy/tabulartoy_25_10k (generated locally)
#
# The training code uses paths relative to the repo root (data/...), so this
# script symlinks the payload directories from the repo's data/ dir onto the
# network volume. Repo code files in data/ (loaders, generators) are untouched.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
DATA_DIR="${DATA_DIR:-$WORKSPACE/data}"
RESULTS_DIR="${RESULTS_DIR:-$WORKSPACE/results}"
LOGS_DIR="${LOGS_DIR:-$WORKSPACE/logs}"
VENV_DIR="${VENV_DIR:-$WORKSPACE/venv}"
# Comma-separated subset of datasets to fetch: celeba,cub,dsprites,tabulartoy
# (default: all). E.g.  DATASETS=celeba,tabulartoy bash scripts/setup_pod.sh
DATASETS="${DATASETS:-all}"
want() { [ "$DATASETS" = "all" ] || [[ ",$DATASETS," == *",$1,"* ]]; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "== setup_pod: repo=$REPO_DIR data=$DATA_DIR results=$RESULTS_DIR =="

mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$LOGS_DIR"

# ---------------------------------------------------------------- venv + deps
if [ ! -d "$VENV_DIR" ]; then
    echo "-- creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
echo "-- installing requirements"
pip install -q -r "$REPO_DIR/requirements.txt"
# Package itself, editable, deps already satisfied above
pip install -q -e "$REPO_DIR" --no-deps

# ------------------------------------------- runpodctl (for AUTOSTOP in sweep)
# Only needed so run_sweep.sh can stop the pod when it finishes. Harmless if
# the key is absent — autostop is opt-in and degrades to a warning.
if ! command -v runpodctl >/dev/null 2>&1; then
    echo "-- installing runpodctl"
    curl -sSL https://cli.runpod.net | bash >/dev/null 2>&1 || \
        echo "!! runpodctl install failed (AUTOSTOP will not work)"
fi
if [ -n "${RUNPOD_API_KEY:-}" ] && command -v runpodctl >/dev/null 2>&1; then
    runpodctl config --apiKey "$RUNPOD_API_KEY" >/dev/null 2>&1 && \
        echo "-- runpodctl configured"
fi

# ------------------------------------------------- symlink payloads into repo
# link <repo/data NAME> -> <DATA_DIR NAME>
link_payload() {
    local repo_path="$REPO_DIR/data/$1"
    local vol_path="$DATA_DIR/$1"
    mkdir -p "$(dirname "$repo_path")" "$(dirname "$vol_path")"
    if [ ! -e "$repo_path" ]; then
        mkdir -p "$vol_path"
        ln -s "$vol_path" "$repo_path"
        echo "-- linked data/$1 -> $vol_path"
    fi
}
link_payload celeba
link_payload dsprites
link_payload TabularToy
link_payload CUB200/CUB_200_2011
link_payload CUB200/class_attr_data_10

# ------------------------------------------------------------------- datasets
cd "$REPO_DIR"

# --- CUB images (Caltech canonical link)
if ! want cub; then
    echo "-- skipping CUB (not in DATASETS=$DATASETS)"
elif [ -z "$(ls -A "$DATA_DIR/CUB200/CUB_200_2011" 2>/dev/null)" ]; then
    echo "-- fetching CUB_200_2011 images (~1.1 GB)"
    wget -q --show-progress -O "$DATA_DIR/CUB200/CUB_200_2011.tgz" \
        "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    tar -xzf "$DATA_DIR/CUB200/CUB_200_2011.tgz" -C "$DATA_DIR/CUB200/" \
        --strip-components=0
    # tgz extracts CUB_200_2011/ + attributes.txt beside it
    rm -f "$DATA_DIR/CUB200/CUB_200_2011.tgz"
else
    echo "-- CUB_200_2011 present, skipping"
fi
# attributes.txt lives at repo data/CUB200/attributes.txt; restore if missing
if [ ! -f "$REPO_DIR/data/CUB200/attributes.txt" ] && \
   [ -f "$DATA_DIR/CUB200/CUB_200_2011/attributes.txt" ]; then
    cp "$DATA_DIR/CUB200/CUB_200_2011/attributes.txt" "$REPO_DIR/data/CUB200/"
fi

# --- CUB processed concept splits (train/val/test pkl, CBM-paper CodaLab bundle)
if ! want cub; then
    :
elif [ -z "$(ls -A "$DATA_DIR/CUB200/class_attr_data_10" 2>/dev/null)" ]; then
    echo "-- fetching CUB class_attr_data_10 splits"
    wget -q --show-progress -O "$DATA_DIR/CUB200/CUB_processed.tar.gz" \
        "https://worksheets.codalab.org/rest/bundles/0xd013a7ba2e88481bbc07e787f73109f5/contents/blob/"
    tar -xzf "$DATA_DIR/CUB200/CUB_processed.tar.gz" -C "$DATA_DIR/CUB200/"
    # bundle extracts class_attr_data_10/ (possibly under CUB_processed/)
    if [ -d "$DATA_DIR/CUB200/CUB_processed/class_attr_data_10" ]; then
        mv "$DATA_DIR/CUB200/CUB_processed/class_attr_data_10/"* \
           "$DATA_DIR/CUB200/class_attr_data_10/"
        rm -rf "$DATA_DIR/CUB200/CUB_processed"
    fi
    rm -f "$DATA_DIR/CUB200/CUB_processed.tar.gz"
else
    echo "-- class_attr_data_10 present, skipping"
fi

# --- CelebA (torchvision layout: $DATA_DIR/celeba/{img_align_celeba,...})
if ! want celeba; then
    echo "-- skipping CelebA (not in DATASETS=$DATASETS)"
elif [ -z "$(ls -A "$DATA_DIR/celeba" 2>/dev/null)" ]; then
    if [ -f "$DATA_DIR/celeba.tar" ]; then
        echo "-- extracting celeba.tar"
        tar -xf "$DATA_DIR/celeba.tar" -C "$DATA_DIR/"
    else
        echo "-- attempting torchvision CelebA download (gdrive; may hit quota)"
        python - <<'EOF' || echo "!! CelebA download failed. Upload celeba.tar to \$DATA_DIR and re-run."
import os
import torchvision
torchvision.datasets.CelebA(
    root=os.environ.get("DATA_DIR", "/workspace/data"),
    split="all", download=True, target_type=["attr"],
)
EOF
    fi
else
    echo "-- celeba present, skipping"
fi

# --- dSprites (raw npz, then generate the dep_0 dataset the configs expect)
if ! want dsprites; then
    echo "-- skipping dSprites (not in DATASETS=$DATASETS)"
elif [ ! -f "$DATA_DIR/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz" ]; then
    echo "-- fetching raw dSprites npz"
    wget -q --show-progress \
        -O "$DATA_DIR/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz" \
        "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
else
    echo "-- raw dSprites npz present, skipping"
fi
if ! want dsprites; then
    :
elif [ ! -f "$DATA_DIR/dsprites/dsprites_dep_0.npz" ]; then
    echo "-- generating dSprites datasets (installs tensorflow-cpu on demand)"
    python -c "import tensorflow" 2>/dev/null || pip install -q tensorflow-cpu
    python data/generate_dsprites_datasets.py
else
    echo "-- generated dSprites datasets present, skipping"
fi

# --- TabularToy (generated; configs expect tabulartoy_25_10k)
if ! want tabulartoy; then
    echo "-- skipping TabularToy (not in DATASETS=$DATASETS)"
elif [ ! -d "$DATA_DIR/TabularToy/tabulartoy_25_10k" ]; then
    echo "-- generating TabularToy (delta=0.25, n=10000)"
    python data/generate_tabulartoy_dataset.py 0.25 10000
else
    echo "-- TabularToy present, skipping"
fi

echo "== setup_pod complete =="
echo "   activate with: source $VENV_DIR/bin/activate"
