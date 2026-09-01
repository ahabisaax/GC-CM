"""
Extract last-epoch state dicts from PyTorch Lightning last.ckpt files and
save them as <model_name>_fold_X_last.pt alongside the existing best .pt files.

Usage (run from project root):
    python experiments/evaluate_models/extract_last_checkpoints.py \
        results/celeba_acbm_39c_lam1/

The script maps last-vN.ckpt files to folds by reading the epoch number and
modification time ordering. It assumes folds were run sequentially (fold_1
first), which is the default behaviour of run_experiments.py.
"""
import os
import sys
import glob
import torch


def extract_last_ckpts(model_dir):
    ckpt_dir = os.path.join(model_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        print(f"No checkpoints/ dir in {model_dir}")
        return

    # Collect all last*.ckpt files, sorted by mtime (oldest = fold_1)
    last_files = sorted(
        glob.glob(os.path.join(ckpt_dir, "last*.ckpt")),
        key=os.path.getmtime,
    )
    if not last_files:
        print(f"No last*.ckpt files in {ckpt_dir}")
        return

    print(f"Found {len(last_files)} last checkpoint(s) in {ckpt_dir}")
    for i, ckpt_path in enumerate(last_files, start=1):
        out_path = os.path.join(
            model_dir, os.path.basename(model_dir) + f"_fold_{i}_last.pt"
        )
        if os.path.exists(out_path):
            print(f"  fold_{i}: already exists → {out_path}")
            continue
        print(f"  fold_{i}: loading {os.path.basename(ckpt_path)} ...", end=" ", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"]
        torch.save(state_dict, out_path)
        print(f"saved → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_last_checkpoints.py <model_dir> [<model_dir2> ...]")
        sys.exit(1)
    for d in sys.argv[1:]:
        # Accept either the model root or a sub-directory with checkpoints
        if os.path.isdir(os.path.join(d, "checkpoints")):
            extract_last_ckpts(d)
        else:
            # Try all subdirs
            for sub in sorted(os.listdir(d)):
                subpath = os.path.join(d, sub)
                if os.path.isdir(os.path.join(subpath, "checkpoints")):
                    extract_last_ckpts(subpath)
