"""
Test one-hot CVL on dSprites CEM/GC-CEM.

Modified CVL step 2:
  Instead of Ridge(y_scalar [N,1] -> R_k [N,m]),
  use Ridge(y_onehot [N,C] -> R_k [N,m]) — multivariate regression.

Compares original vs one-hot R² for each concept.

Run with:
    conda run -n xai-leakage python experiments/evaluate_models/compute_dsprites_cvl_onehot.py
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

from xai_concept_leakage.data.dsprites_auxiliary import dsprites_dataloaders
from experiments.experiment_utils import get_dsprites_extractor_arch
from experiments.evaluate_model import predict_c_y

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAVE_PATH = master + "results/results_dsprites_cvl_onehot.dict"

DS_CEM_FOLDER  = master + "results/dsprites_cem/"
DS_ACEM_FOLDER = master + "results/dsprites_acem_shared_critic/"

MODEL_SPECS = [
    ("cem",    "CEM",    DS_CEM_FOLDER,  "CEM_adam_lr1e-03_bs64_lam_c0.1",                           "lam_c0.1"),
    ("cem",    "CEM",    DS_CEM_FOLDER,  "CEM_adam_lr1e-03_bs64_lam_c0.5",                           "lam_c0.5"),
    ("cem",    "CEM",    DS_CEM_FOLDER,  "CEM_adam_lr1e-03_bs64_lam_c1",                             "lam_c1"),
    ("gc_cem", "GC-CEM", DS_ACEM_FOLDER, "CRCEM_adam_lr5e-05_bs64_lam1_none_lam_c0.1_shared_critic", "lam_c0.1"),
    ("gc_cem", "GC-CEM", DS_ACEM_FOLDER, "CRCEM_adam_lr5e-05_bs64_lam1_none_lam_c0.5_shared_critic", "lam_c0.5"),
    ("gc_cem", "GC-CEM", DS_ACEM_FOLDER, "CRCEM_adam_lr5e-05_bs64_lam1_none_lam_c1_shared_critic",   "lam_c1"),
]

ALL_FOLDS = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
corr = 0
path_dataset = master + f"data/dsprites/dsprites_dep_{corr}.npz"
train_dl, val_dl, test_dl = dsprites_dataloaders(
    path_dataset, val_ratio=0.1, num_workers=0, batch_size=256
)
x2c_extractor = get_dsprites_extractor_arch

# ---------------------------------------------------------------------------
# CVL with one-hot y
# ---------------------------------------------------------------------------
def _to_np(t):
    import torch
    return t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)


def compute_CVL_onehot(c_hat_train, c_hat_test, c_train, c_test, y_train, y_test, alpha=1.0):
    """CVL with one-hot encoded y as predictor in step 2."""
    c_hat_train = np.asarray(c_hat_train)
    c_hat_test  = np.asarray(c_hat_test)
    c_train     = np.asarray(c_train)
    c_test      = np.asarray(c_test)
    y_train     = np.asarray(y_train).ravel()
    y_test      = np.asarray(y_test).ravel()

    classes = np.unique(y_train)
    n_classes = len(classes)

    if n_classes == 2:
        # binary — scalar is equivalent, keep shape [N,1]
        Y_tr = y_train.reshape(-1, 1)
        Y_te = y_test.reshape(-1, 1)
    else:
        # multiclass — one-hot [N, C]
        Y_tr = label_binarize(y_train, classes=classes).astype(float)
        Y_te = label_binarize(y_test,  classes=classes).astype(float)

    K = c_hat_train.shape[1]
    cvl_per_concept    = []
    r2_orig_per_concept = []   # original scalar-y R²
    r2_onehot_per_concept = [] # new one-hot R²

    for k in range(K):
        # Step 1: Ridge(c_k -> c_hat_k) -> residuals
        ridge1 = Ridge(alpha=alpha)
        ridge1.fit(c_train[:, k].reshape(-1, 1), c_hat_train[:, k, :])
        R_train = c_hat_train[:, k, :] - ridge1.predict(c_train[:, k].reshape(-1, 1))
        R_test  = c_hat_test[:, k, :]  - ridge1.predict(c_test[:, k].reshape(-1, 1))

        # Step 2a: original scalar y
        ridge_orig = Ridge(alpha=alpha)
        ridge_orig.fit(y_train.reshape(-1, 1), R_train)
        r2_orig = float(r2_score(R_test, ridge_orig.predict(y_test.reshape(-1, 1)),
                                 multioutput="variance_weighted"))

        # Step 2b: one-hot y (multivariate regression)
        ridge_oh = Ridge(alpha=alpha)
        ridge_oh.fit(Y_tr, R_train)
        r2_oh = float(r2_score(R_test, ridge_oh.predict(Y_te),
                               multioutput="variance_weighted"))

        cvl_per_concept.append(max(0.0, r2_oh))
        r2_orig_per_concept.append(r2_orig)
        r2_onehot_per_concept.append(r2_oh)

        print(f"      concept {k:3d}: r2_orig={r2_orig:.4f}  r2_onehot={r2_oh:.4f}  CVL={max(0,r2_oh):.4f}")

    return {
        "CVL":                   float(np.mean(cvl_per_concept)),
        "CVL_per_concept":       cvl_per_concept,
        "r2_orig_per_concept":   r2_orig_per_concept,
        "r2_onehot_per_concept": r2_onehot_per_concept,
    }


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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

for model_key, model_label, folder, model_name, lam in MODEL_SPECS:
    model_path = os.path.join(folder, model_name)
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
            print(f"  {fold_key}: cached  orig={r['cvl_orig_mean']:.4f}  onehot={r['cvl_onehot_mean']:.4f}")
            continue

        print(f"  {fold_key}: loading embeddings …")
        (c_mix_tr, c_true_tr, y_tr), (c_mix_te, c_true_te, y_te) = get_embeddings(ckpt)
        n_classes = len(np.unique(y_te))
        print(f"    c_mix shape={c_mix_te.shape}  n_classes={n_classes}")

        res = compute_CVL_onehot(c_mix_tr, c_mix_te, c_true_tr, c_true_te, y_tr, y_te)
        r["cvl_orig_mean"]            = float(np.mean(res["r2_orig_per_concept"]))
        r["cvl_onehot_mean"]          = float(np.mean(res["r2_onehot_per_concept"]))
        r["cvl_clipped_mean"]         = float(res["CVL"])
        r["r2_orig_per_concept"]      = res["r2_orig_per_concept"]
        r["r2_onehot_per_concept"]    = res["r2_onehot_per_concept"]
        r["cvl_clipped_per_concept"]  = res["CVL_per_concept"]

        print(f"    orig_mean={r['cvl_orig_mean']:.4f}  onehot_mean={r['cvl_onehot_mean']:.4f}  CVL={r['cvl_clipped_mean']:.4f}")

        results[model_key][lam][fold_key] = r
        joblib.dump(results, SAVE_PATH)

print("\n\n=== SUMMARY ===")
print(f"{'Model':8} {'lam':10}  CVL_orig  CVL_onehot  CVL_clipped")
for model_key, model_label, _, _, lam in MODEL_SPECS:
    folds = results.get(model_key, {}).get(lam, {})
    if not folds:
        continue
    orig    = [r["cvl_orig_mean"]   for r in folds.values() if "cvl_orig_mean"   in r]
    onehot  = [r["cvl_onehot_mean"] for r in folds.values() if "cvl_onehot_mean" in r]
    clipped = [r["cvl_clipped_mean"]for r in folds.values() if "cvl_clipped_mean"in r]
    if orig:
        print(f"{model_label:8} {lam:10}  "
              f"{np.mean(orig):.4f}±{np.std(orig):.4f}  "
              f"{np.mean(onehot):.4f}±{np.std(onehot):.4f}  "
              f"{np.mean(clipped):.4f}±{np.std(clipped):.4f}")

print("\nDone.")
