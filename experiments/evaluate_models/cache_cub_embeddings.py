"""
Re-extract CUB CEM and GC-CEM embeddings with a clean train/test split.

The old cache (cub_cvl_pls_embeddings_fold1_lamc0.1.joblib) used CUB's
model-train-set as CVL-train and model-test-set as CVL-test, causing
distribution shift (model memorised training samples -> step-1 R² negative).

Fix: pool val + test (both fully held-out, same distribution), shuffle,
split 7000 CVL-train / 1000 CVL-test.

Output: results/cub_embeddings_clean_split.joblib
  {'CEM':    (c_mix_tr, c_true_tr, y_tr, c_mix_te, c_true_te, y_te),
   'GC-CEM': (c_mix_tr, c_true_tr, y_tr, c_mix_te, c_true_te, y_te)}
  shapes: c_mix (N, 112, 32), c_true (N, 112), y (N,)

Run:
    conda run -n xai-leakage python experiments/evaluate_models/cache_cub_embeddings.py
"""
import os, sys
import numpy as np
import torch
import joblib

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)
sys.path.insert(0, os.path.join(master, "data/CUB200"))

import data.CUB200.cub_loader as cub
from experiments.evaluate_models.eval_suite import load_predictions

SAVE_PATH = master + "results/cub_embeddings_clean_split.joblib"
N_TRAIN   = 7000
N_TEST    = 1000
SEED      = 42

cfg = {
    "dataset": "cub", "num_workers": 0, "batch_size": 256,
    "root_dir": os.path.join(master, "data/CUB200/"),
    "sampling_percent": 1, "sampling_groups": True,
    "test_subsampling": 1, "weight_loss": True, "train_augment": False,
}

MODELS = {
    "CEM": (master + "results/cub_cem/CEM_adam_lr1e-03_bs256_lam_c0.1/"
            "CEM_adam_lr1e-03_bs256_lam_c0.1_fold_1.pt"),
    "GC-CEM": (master + "results/cub_cem/CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c0.1_shared_critic/"
               "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c0.1_shared_critic_fold_1.pt"),
}

results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

# Load val + test dataloaders
print("Loading CUB dataloaders ...", flush=True)
train_dl, val_dl, test_dl, _, _ = cub.generate_data(
    config=cfg, seed=SEED, output_dataset_vars=True, root_dir=cfg["root_dir"]
)
print(f"  train batches={len(train_dl)}  val batches={len(val_dl)}  test batches={len(test_dl)}")

for model_name, pt_path in MODELS.items():
    if model_name in results:
        c_mix_tr, c_true_tr, y_tr, c_mix_te, c_true_te, y_te = results[model_name]
        print(f"\n{model_name}: already cached  "
              f"train={len(y_tr)}  test={len(y_te)}  [skip]")
        continue

    print(f"\nExtracting {model_name} embeddings ...", flush=True)

    # Extract from val set
    print("  val set ...", flush=True)
    preds_val = load_predictions(pt_path, None, val_dl)
    c_mix_val  = preds_val["c_mix"]   # (N_val, K, m)
    c_true_val = preds_val["c_true"]  # (N_val, K)
    y_val      = preds_val["y_true"]  # (N_val,)
    print(f"  val: {c_mix_val.shape}", flush=True)

    # Extract from test set
    print("  test set ...", flush=True)
    preds_test = load_predictions(pt_path, None, test_dl)
    c_mix_test  = preds_test["c_mix"]
    c_true_test = preds_test["c_true"]
    y_test      = preds_test["y_true"]
    print(f"  test: {c_mix_test.shape}", flush=True)

    # Pool and shuffle
    c_mix_all  = np.concatenate([c_mix_val,  c_mix_test],  axis=0)
    c_true_all = np.concatenate([c_true_val, c_true_test], axis=0)
    y_all      = np.concatenate([y_val,      y_test],      axis=0)
    N_all      = len(y_all)
    print(f"  pooled: {N_all} samples  (val={len(y_val)} + test={len(y_test)})", flush=True)

    rng   = np.random.RandomState(SEED)
    idx   = rng.permutation(N_all)
    n_tr  = min(N_TRAIN, N_all - N_TEST)
    n_te  = N_TEST
    tr_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr + n_te]

    results[model_name] = (
        c_mix_all[tr_idx].astype(np.float32),
        c_true_all[tr_idx].astype(np.float32),
        y_all[tr_idx].astype(np.int64),
        c_mix_all[te_idx].astype(np.float32),
        c_true_all[te_idx].astype(np.float32),
        y_all[te_idx].astype(np.int64),
    )
    print(f"  split: train={n_tr}  test={n_te}", flush=True)
    joblib.dump(results, SAVE_PATH)
    print(f"  saved → {SAVE_PATH}", flush=True)

# Sanity check: step-1 R² should now be positive
print("\nSanity check — step-1 R² on clean split:", flush=True)
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

for model_name in results:
    c_mix_tr, c_true_tr, _, c_mix_te, c_true_te, _ = results[model_name]
    K = c_mix_tr.shape[1]
    r2s = []
    for k in range(K):
        reg = Ridge(alpha=1.0)
        reg.fit(c_true_tr[:, k:k+1], c_mix_tr[:, k, :])
        r2s.append(r2_score(c_mix_te[:, k, :],
                            reg.predict(c_true_te[:, k:k+1]),
                            multioutput="variance_weighted"))
    print(f"  {model_name}: step1_R2 mean={np.mean(r2s):.4f}  std={np.std(r2s):.4f}")

print("\nDone.")
