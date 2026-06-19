"""
CBM-family evaluation suite for TabularToy (cov=0.25, 10k, 3 concepts).

Models evaluated
----------------
  HardCBM  — hard concept bottleneck (lam_c1 only)
  SoftCBM  — soft/joint CBM (lam_c ∈ {0.1, 0.5, 1})
  CRCBM    — critic-regularised CBM (lam_c ∈ {0.1, 0.5, 1})
  ACBM     — adversarial CBM (lam_c ∈ {0.1, 0.5, 1})
  SeqCBM   — sequential CBM (lam_c1 only)

Metrics collected per fold
--------------------------
  task_acc     — test task accuracy
  c_acc        — test concept accuracy
  interv       — random intervention curve loaded from per-fold .npy
  ois_nis_cas  — OIS / NIS / CAS (computed fresh from .pt checkpoint)

Results saved to  results/results_tabulartoy_cbm_suite.dict

Run from project root:
    python experiments/evaluate_models/evaluate_models_tabulartoy_cbm_suite.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

master_folder = os.getcwd().replace("/experiments/evaluate_models", "")
sys.path.insert(0, master_folder)
master_folder = master_folder + "/"

import numpy as np
import joblib

from xai_concept_leakage.data.tabulartoy_auxiliary import TT_dataloaders
from experiments.experiment_utils import get_tabulartoy_extractor_arch
from experiments.evaluate_models.eval_suite import ALL_FOLDS, run_ois_nis_cas

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRIAL_RUN = False
FOLDS     = ["fold_1"] if TRIAL_RUN else ALL_FOLDS

RERUN_TASK    = False
RERUN_INTERV  = False
RERUN_OIS     = False

TT_MAIN_FOLDER = master_folder + "results/tabulartoy_25_10k_models/"
TT_ACBM_FOLDER = master_folder + "results/tabulartoy_25_10k_models_acbm_shared_critic/"
SAVE_PATH      = master_folder + "results/results_tabulartoy_cbm_suite.dict"

# (label, folder, model_dir_name, lam_c_key)
MODEL_SPECS = [
    ("hard_cbm", TT_MAIN_FOLDER, "HardCBM_adam_lr0.05_bs64_lam_c1",                       "lam_c1"),
    ("soft_cbm", TT_MAIN_FOLDER, "SoftCBM_adam_lr0.05_bs64_lam_c0.1",                     "lam_c0.1"),
    ("soft_cbm", TT_MAIN_FOLDER, "SoftCBM_adam_lr0.05_bs64_lam_c0.5",                     "lam_c0.5"),
    ("soft_cbm", TT_MAIN_FOLDER, "SoftCBM_adam_lr0.05_bs64_lam_c1",                       "lam_c1"),
    ("crcbm",    TT_MAIN_FOLDER, "CRCBM_adam_lr0.05_bs64_lam1_none_lam_c0.1",             "lam_c0.1"),
    ("crcbm",    TT_MAIN_FOLDER, "CRCBM_adam_lr0.05_bs64_lam1_none_lam_c0.5",             "lam_c0.5"),
    ("crcbm",    TT_MAIN_FOLDER, "CRCBM_adam_lr0.05_bs64_lam1_none_lam_c1",               "lam_c1"),
    ("acbm",     TT_ACBM_FOLDER, "ACBM_adam_lr0.05_bs64_lam1_none_lam_c0.1_shared_critic","lam_c0.1"),
    ("acbm",     TT_ACBM_FOLDER, "ACBM_adam_lr0.05_bs64_lam1_none_lam_c0.5_shared_critic","lam_c0.5"),
    ("acbm",     TT_ACBM_FOLDER, "ACBM_adam_lr0.05_bs64_lam1_none_lam_c1_shared_critic",  "lam_c1"),
    ("seq_cbm",  TT_ACBM_FOLDER, "SeqCBM_adam_lr0.05_bs64_lam_c1",                        "lam_c1"),
]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data_folder = master_folder + "data/TabularToy/"
save_folder = data_folder + "tabulartoy_25_10k/"
train_dl, val_dl, test_dl = TT_dataloaders(
    save_folder, considered_concepts=["0", "1", "2"], c_logits=False, num_workers=0
)
x2c_extractor = get_tabulartoy_extractor_arch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _interv_npy(model_path, model_name, fold_idx_0):
    """Load random intervention curve for fold (0-indexed)."""
    npy = os.path.join(
        model_path,
        f"test_acc_y_random_group_level_True_use_prior_False_ints_{model_name}_fold_{fold_idx_0}.npy"
    )
    if os.path.exists(npy):
        return np.load(npy).tolist()
    return None


def _split_results(model_path, model_name, split_idx):
    """Load per-split results.joblib (split_idx is 0-indexed)."""
    path = os.path.join(model_path, f"{model_name}_split_{split_idx}_results.joblib")
    return joblib.load(path) if os.path.exists(path) else {}


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------
results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

for label, folder, model_name, lam_c in MODEL_SPECS:
    model_path = os.path.join(folder, model_name)
    if not os.path.isdir(model_path):
        print(f"  MISSING: {model_path}")
        continue

    results.setdefault(label, {}).setdefault(lam_c, {})
    print(f"\n{label} | {lam_c}")

    for fold_key in FOLDS:                       # "fold_1" .. "fold_5"
        fold_1idx  = int(fold_key.split("_")[1]) # 1-indexed
        fold_0idx  = fold_1idx - 1               # 0-indexed (npy / split)

        r    = results[label][lam_c].get(fold_key, {})
        todo = {
            "task":  "task_acc"    not in r or RERUN_TASK,
            "interv":"interv"      not in r or RERUN_INTERV,
            "ois":   "ois_nis_cas" not in r or RERUN_OIS,
        }

        # task / concept accuracy
        if todo["task"]:
            sr = _split_results(model_path, model_name, fold_0idx)
            if sr:
                r["task_acc"] = float(sr.get("test_acc_y", float("nan")))
                r["c_acc"]    = float(sr.get("test_acc_c", float("nan")))

        # intervention curve
        if todo["interv"]:
            curve = _interv_npy(model_path, model_name, fold_0idx)
            if curve is not None:
                r["interv"] = {"random": [curve]}

        # OIS / NIS / CAS
        ckpt = os.path.join(model_path, f"{model_name}_{fold_key}.pt")
        if todo["ois"] and os.path.exists(ckpt):
            print(f"  {fold_key}: running OIS/NIS/CAS …")
            r["ois_nis_cas"] = run_ois_nis_cas(ckpt, x2c_extractor, train_dl, test_dl)
        elif not os.path.exists(ckpt):
            print(f"  {fold_key}: checkpoint missing — {ckpt}")

        task  = r.get("task_acc", float("nan"))
        c_acc = r.get("c_acc",    float("nan"))
        ois   = r.get("ois_nis_cas", {})
        iv    = r.get("interv", {}).get("random", [[]])[0]
        gain  = (iv[-1] - iv[0]) if iv else float("nan")
        print(f"  {fold_key}  task={task:.4f}  c_acc={c_acc:.4f}  "
              f"interv_gain={gain:+.4f}  "
              f"OIS={ois.get('ois', float('nan')):.4f}  "
              f"NIS={ois.get('nis', float('nan')):.4f}  "
              f"CAS={ois.get('cas', float('nan')):.4f}")

        results[label][lam_c][fold_key] = r

    joblib.dump(results, SAVE_PATH)
    print(f"  Saved → {SAVE_PATH}")

print("\nDone.")
