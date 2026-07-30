"""
CVL with one-hot encoded y for TabularToy CEM/GC-CEM.

TT is binary (C=2) so scalar and one-hot are equivalent — this script
verifies that empirically and stores results in the same format as
the dSprites one-hot script for consistency.

Run with:
    conda run -n xai-leakage python experiments/evaluate_models/compute_tabulartoy_cvl_onehot.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)

import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import label_binarize

from xai_concept_leakage.data.tabulartoy_auxiliary import TT_dataloaders
from experiments.experiment_utils import get_tabulartoy_extractor_arch
from experiments.evaluate_model import predict_c_y

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAVE_PATH = master + "results/results_tabulartoy_cvl_onehot.dict"

TT_ACEM_FOLDER = master + "results/tabulartoy_25_10k_models_acem_shared_critic/"

MODEL_SPECS = [
    ("cem",    "CEM",    TT_ACEM_FOLDER, "CEM_adam_lr0.05_bs64_lam_c0.1",                          "lam_c0.1"),
    ("cem",    "CEM",    TT_ACEM_FOLDER, "CEM_adam_lr0.05_bs64_lam_c0.5",                          "lam_c0.5"),
    ("cem",    "CEM",    TT_ACEM_FOLDER, "CEM_adam_lr0.05_bs64_lam_c1",                            "lam_c1"),
    ("gc_cem", "GC-CEM", TT_ACEM_FOLDER, "ACEM_adam_lr1e-03_bs64_lam1_none_lam_c0.1_shared_critic","lam_c0.1"),
    ("gc_cem", "GC-CEM", TT_ACEM_FOLDER, "ACEM_adam_lr1e-03_bs64_lam1_none_lam_c0.5_shared_critic","lam_c0.5"),
    ("gc_cem", "GC-CEM", TT_ACEM_FOLDER, "ACEM_adam_lr1e-03_bs64_lam1_none_lam_c1_shared_critic",  "lam_c1"),
]

ALL_FOLDS = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
save_folder = master + "data/TabularToy/tabulartoy_25_10k/"
train_dl, val_dl, test_dl = TT_dataloaders(
    save_folder, considered_concepts=["0", "1", "2"], c_logits=False, num_workers=0
)
x2c_extractor = get_tabulartoy_extractor_arch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_np(t):
    import torch
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def get_embeddings(ckpt_path):
    out_tr = predict_c_y(train_dl, ckpt_path, x2c_extractor,
                         c_sem_out=True, soft_prob_out=True, vec_emb_out=False)
    out_te = predict_c_y(test_dl,  ckpt_path, x2c_extractor,
                         c_sem_out=True, soft_prob_out=True, vec_emb_out=False)

    def unpack(out):
        c_sem, c_prob, c_true, y_pred, y_true = out[:5]
        c_true = _to_np(c_true)
        y_true = _to_np(y_true).ravel()
        n_k    = c_true.shape[1]
        emb    = _to_np(c_sem).reshape(-1, n_k, _to_np(c_sem).shape[1] // n_k)
        return emb, c_true, y_true

    return unpack(out_tr), unpack(out_te)


def compute_CVL_onehot(c_hat_tr, c_hat_te, c_tr, c_te, y_tr, y_te, alpha=1.0):
    c_hat_tr = np.asarray(c_hat_tr)
    c_hat_te = np.asarray(c_hat_te)
    c_tr     = np.asarray(c_tr)
    c_te     = np.asarray(c_te)
    y_tr     = np.asarray(y_tr).ravel()
    y_te     = np.asarray(y_te).ravel()

    classes   = np.unique(y_tr)
    n_classes = len(classes)
    print(f"      n_classes={n_classes}")

    if n_classes == 2:
        Y_tr = y_tr.reshape(-1, 1).astype(float)
        Y_te = y_te.reshape(-1, 1).astype(float)
    else:
        Y_tr = label_binarize(y_tr, classes=classes).astype(float)
        Y_te = label_binarize(y_te, classes=classes).astype(float)

    K = c_hat_tr.shape[1]
    r2_orig, r2_onehot, r2_clipped = [], [], []

    for k in range(K):
        ridge1 = Ridge(alpha=alpha)
        ridge1.fit(c_tr[:, k].reshape(-1, 1), c_hat_tr[:, k, :])
        R_tr = c_hat_tr[:, k, :] - ridge1.predict(c_tr[:, k].reshape(-1, 1))
        R_te = c_hat_te[:, k, :] - ridge1.predict(c_te[:, k].reshape(-1, 1))

        # scalar y
        ridge_s = Ridge(alpha=alpha)
        ridge_s.fit(y_tr.reshape(-1, 1), R_tr)
        r2_s = float(r2_score(R_te, ridge_s.predict(y_te.reshape(-1, 1)),
                               multioutput="variance_weighted"))

        # one-hot y
        ridge_h = Ridge(alpha=alpha)
        ridge_h.fit(Y_tr, R_tr)
        r2_h = float(r2_score(R_te, ridge_h.predict(Y_te),
                               multioutput="variance_weighted"))

        r2_orig.append(r2_s)
        r2_onehot.append(r2_h)
        r2_clipped.append(max(0.0, r2_h))
        print(f"      concept {k}: scalar={r2_s:.4f}  onehot={r2_h:.4f}")

    return {
        "CVL":                    float(np.mean(r2_clipped)),
        "CVL_per_concept":        r2_clipped,
        "r2_orig_per_concept":    r2_orig,
        "r2_onehot_per_concept":  r2_onehot,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

for model_key, model_label, folder, model_name, lam in MODEL_SPECS:
    model_path = os.path.join(folder, model_name)
    if not os.path.exists(model_path):
        print(f"MISSING: {model_path}")
        continue
    results.setdefault(model_key, {}).setdefault(lam, {})
    print(f"\n{'='*60}")
    print(f"{model_label}  {lam}")

    for fold_key in ALL_FOLDS:
        ckpt = os.path.join(model_path, f"{model_name}_{fold_key}.pt")
        if not os.path.exists(ckpt):
            print(f"  {fold_key}: checkpoint missing")
            continue

        r = results[model_key][lam].get(fold_key, {})
        if "cvl_onehot_mean" in r:
            print(f"  {fold_key}: cached  scalar={r['cvl_orig_mean']:.4f}  onehot={r['cvl_onehot_mean']:.4f}")
            continue

        print(f"  {fold_key}: loading embeddings ...")
        (emb_tr, c_tr, y_tr), (emb_te, c_te, y_te) = get_embeddings(ckpt)
        print(f"    emb shape={emb_te.shape}")

        res = compute_CVL_onehot(emb_tr, emb_te, c_tr, c_te, y_tr, y_te)
        r["cvl_orig_mean"]   = float(np.mean(res["r2_orig_per_concept"]))
        r["cvl_onehot_mean"] = float(np.mean(res["r2_onehot_per_concept"]))
        r["cvl_clipped_mean"]= float(res["CVL"])
        r["r2_orig_per_concept"]   = res["r2_orig_per_concept"]
        r["r2_onehot_per_concept"] = res["r2_onehot_per_concept"]
        r["cvl_clipped_per_concept"] = res["CVL_per_concept"]

        print(f"    scalar={r['cvl_orig_mean']:.4f}  onehot={r['cvl_onehot_mean']:.4f}  CVL={r['cvl_clipped_mean']:.4f}")
        results[model_key][lam][fold_key] = r
        joblib.dump(results, SAVE_PATH)

print("\n\n=== SUMMARY ===")
print(f"{'Model':8} {'lam':10}  CVL_scalar  CVL_onehot  CVL_clipped")
for model_key, model_label, _, _, lam in MODEL_SPECS:
    folds = results.get(model_key, {}).get(lam, {})
    if not folds:
        continue
    orig    = [r["cvl_orig_mean"]    for r in folds.values() if "cvl_orig_mean"    in r]
    onehot  = [r["cvl_onehot_mean"]  for r in folds.values() if "cvl_onehot_mean"  in r]
    clipped = [r["cvl_clipped_mean"] for r in folds.values() if "cvl_clipped_mean" in r]
    if orig:
        print(f"{model_label:8} {lam:10}  "
              f"{np.mean(orig):.4f}±{np.std(orig):.4f}  "
              f"{np.mean(onehot):.4f}±{np.std(onehot):.4f}  "
              f"{np.mean(clipped):.4f}±{np.std(clipped):.4f}")
print("\nDone.")
