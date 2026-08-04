"""
Extract CUB embeddings from one or two model checkpoints and cache them
for CVL/ICVL analysis.

Uses the clean-split protocol: pool val+test (both held-out), shuffle,
split 5992/1000.

Usage:
    python experiments/evaluate_models/cache_cub_embeddings_single.py \
        --gcem_ckpt  results/cub_gcem_emb16/GCEM-emb16-lam01/GCEM-emb16-lam01_fold_1.pt \
        --cem_ckpt   results/cub_cem/CEM_adam_lr1e-03_bs256_lam_c0.1/CEM_adam_lr1e-03_bs256_lam_c0.1_fold_1.pt \
        --output     results/cub_embeddings_gcem_emb16.joblib \
        --data_root  data/CUB200/
"""
import argparse, os, sys
import numpy as np
import joblib

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)
sys.path.insert(0, os.path.join(master, "data/CUB200"))

import data.CUB200.cub_loader as cub
from experiments.evaluate_models.eval_suite import load_predictions
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

N_TRAIN = 5992
N_TEST  = 1000
SEED    = 42


def extract_model(pt_path, val_dl, test_dl, label):
    print(f"\nExtracting {label} embeddings from {pt_path} ...", flush=True)
    preds_val  = load_predictions(pt_path, None, val_dl)
    preds_test = load_predictions(pt_path, None, test_dl)

    c_mix  = np.concatenate([preds_val["c_mix"],   preds_test["c_mix"]],  axis=0)
    c_true = np.concatenate([preds_val["c_true"],  preds_test["c_true"]], axis=0)
    y      = np.concatenate([preds_val["y_true"],  preds_test["y_true"]], axis=0)

    N = len(y)
    rng    = np.random.RandomState(SEED)
    idx    = rng.permutation(N)
    n_tr   = min(N_TRAIN, N - N_TEST)
    tr_idx = idx[:n_tr]
    te_idx = idx[n_tr:n_tr + N_TEST]

    print(f"  pooled {N}  →  train={n_tr}  test={N_TEST}", flush=True)
    print(f"  c_mix shape: {c_mix.shape}", flush=True)
    return (
        c_mix[tr_idx].astype(np.float32),
        c_true[tr_idx].astype(np.float32),
        y[tr_idx].astype(np.int64),
        c_mix[te_idx].astype(np.float32),
        c_true[te_idx].astype(np.float32),
        y[te_idx].astype(np.int64),
    )


def step1_r2(cache_entry, label):
    c_mix_tr, c_true_tr, _, c_mix_te, c_true_te, _ = cache_entry
    K  = c_mix_tr.shape[1]
    r2 = []
    for k in range(K):
        reg = Ridge(alpha=1.0)
        reg.fit(c_true_tr[:, k:k+1], c_mix_tr[:, k, :])
        r2.append(r2_score(c_mix_te[:, k, :],
                           reg.predict(c_true_te[:, k:k+1]),
                           multioutput="variance_weighted"))
    print(f"  {label}: step1_R2 mean={np.mean(r2):.4f}  std={np.std(r2):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcem_ckpt",  required=True,
                    help="Path to GC-CEM .pt checkpoint")
    ap.add_argument("--cem_ckpt",   default=None,
                    help="Path to CEM .pt checkpoint (optional; skip if not provided)")
    ap.add_argument("--output",     required=True,
                    help="Output .joblib path for embedding cache")
    ap.add_argument("--data_root",  default="data/CUB200/",
                    help="CUB200 data root directory")
    args = ap.parse_args()

    data_root = args.data_root if os.path.isabs(args.data_root) \
                else os.path.join(master, args.data_root)
    output    = args.output if os.path.isabs(args.output) \
                else os.path.join(master, args.output)

    cfg = {
        "dataset": "cub", "num_workers": 0, "batch_size": 256,
        "root_dir": data_root,
        "sampling_percent": 1, "sampling_groups": True,
        "test_subsampling": 1, "weight_loss": True, "train_augment": False,
    }

    print("Loading CUB dataloaders ...", flush=True)
    _, val_dl, test_dl, _, _ = cub.generate_data(
        config=cfg, seed=SEED, output_dataset_vars=True, root_dir=data_root
    )
    print(f"  val batches={len(val_dl)}  test batches={len(test_dl)}", flush=True)

    cache = joblib.load(output) if os.path.exists(output) else {}

    if "GC-CEM" not in cache:
        cache["GC-CEM"] = extract_model(args.gcem_ckpt, val_dl, test_dl, "GC-CEM")
        joblib.dump(cache, output)
        print(f"  saved → {output}", flush=True)
    else:
        print("GC-CEM already cached — skipping extraction.")

    if args.cem_ckpt and "CEM" not in cache:
        cache["CEM"] = extract_model(args.cem_ckpt, val_dl, test_dl, "CEM")
        joblib.dump(cache, output)
        print(f"  saved → {output}", flush=True)
    elif args.cem_ckpt is None:
        print("No --cem_ckpt provided; CEM not in cache.")

    print("\nSanity check — step-1 R²:")
    for label in cache:
        step1_r2(cache[label], label)

    print("\nDone.")


if __name__ == "__main__":
    main()
