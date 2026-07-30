"""
CVL with one-hot encoded y for CUB-200 CEM/GC-CEM models.

CUB has 200 classes; the original scalar-y predictor is meaningless
for multiclass. This script one-hot encodes y → [N, 200] and runs
multivariate Ridge regression for step 2 of CVL.

Also computes ICVL (unchanged — uses binary concept labels, not y).

Results saved to results/results_cub_cvl_onehot.dict

Run with:
    conda run -n xai-leakage python experiments/evaluate_models/compute_cub_cvl_onehot.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)
sys.path.insert(0, master + "data/CUB200")

import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import label_binarize
import data.CUB200.cub_loader as cub_data_module
from experiments.evaluate_model import predict_c_y

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RERUN     = False
SAVE_PATH = master + "results/results_cub_cvl_onehot.dict"
CUB_CEM_FOLDER = master + "results/cub_cem/"

MODEL_SPECS = [
    ("cem",    "CEM",    "CEM_adam_lr1e-03_bs256_lam_c0.1",                           "lam_c0.1"),
    ("cem",    "CEM",    "CEM_adam_lr1e-03_bs256_lam_c0.5",                           "lam_c0.5"),
    ("cem",    "CEM",    "CEM_adam_lr1e-03_bs256_lam_c1",                             "lam_c1"),
    ("gc_cem", "GC-CEM", "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c0.1_shared_critic", "lam_c0.1"),
    ("gc_cem", "GC-CEM", "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c0.5_shared_critic", "lam_c0.5"),
    ("gc_cem", "GC-CEM", "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c1_shared_critic",   "lam_c1"),
]

ALL_FOLDS = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
dataset_config = {
    "dataset": "cub",
    "root_dir": master + "data/CUB200/",
    "batch_size": 256,
    "num_workers": 0,
    "sampling_percent": 1,
    "sampling_groups": True,
    "test_subsampling": 1,
    "weight_loss": True,
    "train_augment": False,
}
train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = (
    cub_data_module.generate_data(
        config=dataset_config, seed=42,
        output_dataset_vars=True,
        root_dir=dataset_config["root_dir"],
    )
)
print(f"CUB: n_concepts={n_concepts}, n_tasks={n_tasks}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_np(t):
    import torch
    if isinstance(t, np.ndarray):
        return t
    return t.detach().cpu().numpy()


def get_embeddings(ckpt_path):
    out_tr = predict_c_y(
        dl=train_dl, model_path=ckpt_path, x2c_extractor=None,
        c_sem_out=True, soft_prob_out=True, vec_emb_out=False,
    )
    out_te = predict_c_y(
        dl=test_dl, model_path=ckpt_path, x2c_extractor=None,
        c_sem_out=True, soft_prob_out=True, vec_emb_out=False,
    )

    def unpack(out):
        c_sem, c_prob, c_true, y_pred, y_true = out[:5]
        c_true = _to_np(c_true)
        y_true = _to_np(y_true).ravel()
        n_k    = c_true.shape[1]
        emb    = _to_np(c_sem).reshape(-1, n_k, _to_np(c_sem).shape[1] // n_k)
        return emb, c_true, y_true

    return unpack(out_tr), unpack(out_te)


def compute_CVL_onehot(c_hat_tr, c_hat_te, c_tr, c_te, y_tr, y_te, alpha=1.0):
    """CVL with one-hot y predictor; returns per-concept and global stats."""
    c_hat_tr = np.asarray(c_hat_tr)
    c_hat_te = np.asarray(c_hat_te)
    c_tr     = np.asarray(c_tr)
    c_te     = np.asarray(c_te)
    y_tr     = np.asarray(y_tr).ravel()
    y_te     = np.asarray(y_te).ravel()

    classes   = np.unique(y_tr)
    n_classes = len(classes)
    print(f"      n_classes={n_classes}, one-hot shape: ({len(y_tr)}, {n_classes})")

    if n_classes == 2:
        Y_tr = y_tr.reshape(-1, 1)
        Y_te = y_te.reshape(-1, 1)
    else:
        Y_tr = label_binarize(y_tr, classes=classes).astype(float)
        Y_te = label_binarize(y_te, classes=classes).astype(float)

    K = c_hat_tr.shape[1]
    r2_raw     = []
    r2_clipped = []

    for k in range(K):
        ridge1 = Ridge(alpha=alpha)
        ridge1.fit(c_tr[:, k].reshape(-1, 1), c_hat_tr[:, k, :])
        R_tr = c_hat_tr[:, k, :] - ridge1.predict(c_tr[:, k].reshape(-1, 1))
        R_te = c_hat_te[:, k, :] - ridge1.predict(c_te[:, k].reshape(-1, 1))

        ridge2 = Ridge(alpha=alpha)
        ridge2.fit(Y_tr, R_tr)
        r2 = float(r2_score(R_te, ridge2.predict(Y_te), multioutput="variance_weighted"))

        r2_raw.append(r2)
        r2_clipped.append(max(0.0, r2))

    return {
        "r2_raw":         r2_raw,
        "r2_clipped":     r2_clipped,
        "CVL_per_concept": r2_clipped,
        "CVL":            float(np.mean(r2_clipped)),
    }


def compute_ICVL(c_hat_tr, c_hat_te, c_tr, c_te, alpha=1.0):
    """ICVL: Ridge(c_i → R_j) for all pairs i≠j."""
    c_hat_tr = np.asarray(c_hat_tr)
    c_hat_te = np.asarray(c_hat_te)
    c_tr     = np.asarray(c_tr)
    c_te     = np.asarray(c_te)

    K = c_hat_tr.shape[1]
    r2_mat = np.zeros((K, K))

    for j in range(K):
        # Residual R_j after removing c_j
        ridge1 = Ridge(alpha=alpha)
        ridge1.fit(c_tr[:, j].reshape(-1, 1), c_hat_tr[:, j, :])
        R_tr = c_hat_tr[:, j, :] - ridge1.predict(c_tr[:, j].reshape(-1, 1))
        R_te = c_hat_te[:, j, :] - ridge1.predict(c_te[:, j].reshape(-1, 1))

        for i in range(K):
            if i == j:
                continue
            ridge2 = Ridge(alpha=alpha)
            ridge2.fit(c_tr[:, i].reshape(-1, 1), R_tr)
            r2 = float(r2_score(R_te, ridge2.predict(c_te[:, i].reshape(-1, 1)),
                                 multioutput="variance_weighted"))
            r2_mat[i, j] = r2

    mask = ~np.eye(K, dtype=bool)
    r2_off = r2_mat[mask]
    return {
        "r2_matrix":   r2_mat.tolist(),
        "r2_raw_mean": float(np.mean(r2_off)),
        "r2_raw_max":  float(np.max(r2_off)),
        "ICVL":        float(np.mean(np.maximum(0.0, r2_off))),
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

for model_key, model_label, model_name, lam in MODEL_SPECS:
    model_path = os.path.join(CUB_CEM_FOLDER, model_name)
    results.setdefault(model_key, {}).setdefault(lam, {})
    print(f"\n{'='*60}")
    print(f"{model_label}  {lam}")

    for fold_key in ALL_FOLDS:
        ckpt = os.path.join(model_path, f"{model_name}_{fold_key}.pt")
        if not os.path.exists(ckpt):
            print(f"  {fold_key}: checkpoint missing")
            continue

        r = results[model_key][lam].get(fold_key, {})
        need_cvl  = "cvl_onehot_mean" not in r or RERUN
        need_icvl = "icvl_mean"       not in r or RERUN

        if not (need_cvl or need_icvl):
            print(f"  {fold_key}: cached  CVL={r['cvl_onehot_mean']:.4f}  ICVL={r['icvl_mean']:.6f}")
            continue

        print(f"  {fold_key}: loading embeddings …")
        (c_mix_tr, c_true_tr, y_tr), (c_mix_te, c_true_te, y_te) = get_embeddings(ckpt)
        print(f"    c_mix={c_mix_te.shape}  y_classes={len(np.unique(y_te))}")

        if need_cvl:
            print(f"    computing CVL (one-hot) …")
            cvl = compute_CVL_onehot(c_mix_tr, c_mix_te, c_true_tr, c_true_te, y_tr, y_te)
            r["cvl_r2_raw"]      = cvl["r2_raw"]
            r["cvl_r2_clipped"]  = cvl["r2_clipped"]
            r["cvl_onehot_mean"] = float(np.mean(cvl["r2_raw"]))
            r["cvl_mean"]        = cvl["CVL"]
            print(f"    CVL raw mean={r['cvl_onehot_mean']:.4f}  CVL clipped={cvl['CVL']:.4f}")

        if need_icvl:
            print(f"    computing ICVL …")
            icvl = compute_ICVL(c_mix_tr, c_mix_te, c_true_tr, c_true_te)
            r["icvl_r2_raw"]   = icvl["r2_raw_mean"]
            r["icvl_r2_max"]   = icvl["r2_raw_max"]
            r["icvl_mean"]     = icvl["ICVL"]
            print(f"    ICVL raw mean={icvl['r2_raw_mean']:.6f}  ICVL clipped={icvl['ICVL']:.6f}")

        results[model_key][lam][fold_key] = r
        joblib.dump(results, SAVE_PATH)
        print(f"    Saved.")

print("\n\n=== SUMMARY ===")
print(f"{'Model':8} {'lam':10}  CVL_raw   CVL      ICVL_raw  ICVL")
for model_key, model_label, model_name, lam in MODEL_SPECS:
    folds = results.get(model_key, {}).get(lam, {})
    if not folds:
        continue
    cvl_raw  = [r["cvl_onehot_mean"] for r in folds.values() if "cvl_onehot_mean" in r]
    cvl      = [r["cvl_mean"]        for r in folds.values() if "cvl_mean"         in r]
    icvl_raw = [r["icvl_r2_raw"]     for r in folds.values() if "icvl_r2_raw"      in r]
    icvl     = [r["icvl_mean"]       for r in folds.values() if "icvl_mean"         in r]
    if cvl_raw:
        print(f"{model_label:8} {lam:10}  "
              f"{np.mean(cvl_raw):.4f}±{np.std(cvl_raw):.4f}  "
              f"{np.mean(cvl):.4f}±{np.std(cvl):.4f}  "
              f"{np.mean(icvl_raw):.4f}±{np.std(icvl_raw):.4f}  "
              f"{np.mean(icvl):.4f}±{np.std(icvl):.4f}")

print("\nDone.")
