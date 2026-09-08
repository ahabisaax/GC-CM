#!/bin/bash
# Fetch AwA2 into ~/Scratch/xai-crcbm/data/. RUN THIS ON A LOGIN NODE, not via
# qsub — compute nodes typically have no outbound internet.
#
#   bash experiments/fetch_awa2_hpc.sh
#
# Downloads ~13 GB and extracts to ~26 GB total. Idempotent: skips the download
# if the archive is present and skips extraction if JPEGImages already exists.
set -euo pipefail

DATA_DIR="${DATA_DIR:-$HOME/Scratch/xai-crcbm/data}"
AWA="$DATA_DIR/AwA2"
URL="https://cvml.ista.ac.at/AwA2/AwA2-data.zip"

mkdir -p "$AWA"

if [ -d "$AWA/Animals_with_Attributes2/JPEGImages" ]; then
    echo "AwA2 already extracted at $AWA/Animals_with_Attributes2 — nothing to do"
    exit 0
fi

if [ ! -f "$AWA/AwA2-data.zip" ]; then
    echo "Downloading AwA2 (~13 GB) to $AWA ..."
    wget -c --progress=dot:giga -O "$AWA/AwA2-data.zip" "$URL"
else
    echo "Archive already present; resuming/skipping download"
    wget -c --progress=dot:giga -O "$AWA/AwA2-data.zip" "$URL" || true
fi

echo "Extracting ..."
unzip -q -o "$AWA/AwA2-data.zip" -d "$AWA/"

BASE="$AWA/Animals_with_Attributes2"
echo "--- verifying layout ---"
for f in classes.txt predicates.txt predicate-matrix-binary.txt; do
    [ -f "$BASE/$f" ] && echo "  OK   $f" || { echo "  MISSING $f"; exit 1; }
done
n_cls=$(ls "$BASE/JPEGImages" | wc -l)
n_img=$(find "$BASE/JPEGImages" -name '*.jpg' | wc -l)
echo "  classes dirs: $n_cls (expect 50)"
echo "  images:       $n_img (expect ~37322)"
echo "  matrix shape: $(awk 'NR==1{print NF" cols"} END{print NR" rows"}' "$BASE/predicate-matrix-binary.txt" | tr '\n' ' ') (expect 85 cols, 50 rows)"

echo
echo "Done. Archive kept at $AWA/AwA2-data.zip — delete it to reclaim ~13 GB:"
echo "  rm $AWA/AwA2-data.zip"
