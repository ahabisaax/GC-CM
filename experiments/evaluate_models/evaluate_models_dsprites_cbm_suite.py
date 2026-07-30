"""
CBM-family evaluation suite for dSprites (dep=0, 5 concepts).

Models evaluated
----------------
  HardCBM  — lam_c1 only
  SoftCBM  — lam_c ∈ {0.1, 0.5, 1}
  GC-CBM   — lam_c ∈ {0.1, 0.5, 1}  (AdversarialConceptBottleneckModel, shared_critic=True)
  SeqCBM   — lam_c1 only

NOTE: GC-CBM uses models from results/dsprites_ACBM_shared_critic/ (folder contains
CRCBM_*_shared_critic entries but their saved config has architecture=
AdversarialConceptBottleneckModel, shared_critic=True). Do NOT use
results/dsprites_dep_0_models/CRCBM_* — those are CriticRegularisedConceptBottleneckModel
(no shared critic) and are superseded.

Metrics per fold: task_acc, c_acc, interv (random curve), ois_nis_cas.

Results saved to  results/results_dsprites_cbm_suite.dict

Run from project root:
    python experiments/evaluate_models/evaluate_models_dsprites_cbm_suite.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

master_folder = os.getcwd().replace("/experiments/evaluate_models", "")
sys.path.insert(0, master_folder)
master_folder = master_folder + "/"

import numpy as np
import joblib

from xai_concept_leakage.data.dsprites_auxiliary import dsprites_dataloaders
from experiments.experiment_utils import get_dsprites_extractor_arch
from experiments.evaluate_models.eval_suite import ALL_FOLDS, run_ois_nis_cas

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRIAL_RUN = False
FOLDS     = ["fold_1"] if TRIAL_RUN else ALL_FOLDS

RERUN_TASK   = False
RERUN_INTERV = False
RERUN_OIS    = False

DS_MAIN_FOLDER   = master_folder + "results/dsprites_dep_0_models/"
DS_ACBM_FOLDER   = master_folder + "results/dsprites_ACBM_shared_critic/"
DS_SEQ_FOLDER    = master_folder + "results/dsprites_sequential_acbm/"
SAVE_PATH        = master_folder + "results/results_dsprites_cbm_suite.dict"

# GC-CBM entries: AdversarialConceptBottleneckModel with shared_critic=True.
# Folder name contains "CRCBM" for historical reasons but architecture is NOT
# CriticRegularisedConceptBottleneckModel — verify with *_experiment_config.joblib.
MODEL_SPECS = [
    ("hard_cbm", DS_MAIN_FOLDER, "HardCBM_adam_lr1e-04_bs64_lam_c1",                                   "lam_c1"),
    ("soft_cbm", DS_MAIN_FOLDER, "SoftCBM_adam_lr1e-04_bs64_lam_c0.1",                                 "lam_c0.1"),
    ("soft_cbm", DS_MAIN_FOLDER, "SoftCBM_adam_lr1e-04_bs64_lam_c0.5",                                 "lam_c0.5"),
    ("soft_cbm", DS_MAIN_FOLDER, "SoftCBM_adam_lr1e-04_bs64_lam_c1",                                   "lam_c1"),
    ("gc_cbm",   DS_ACBM_FOLDER, "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c0.1_shared_critic",           "lam_c0.1"),
    ("gc_cbm",   DS_ACBM_FOLDER, "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c0.5_shared_critic",           "lam_c0.5"),
    ("gc_cbm",   DS_ACBM_FOLDER, "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c1_shared_critic",             "lam_c1"),
    ("seq_cbm",  DS_SEQ_FOLDER,  "SeqCBM_adam_lr1e-04_bs64_lam_c1",                                    "lam_c1"),
]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
corr = 0
path_dataset = master_folder + f"data/dsprites/dsprites_dep_{corr}.npz"
train_dl, val_dl, test_dl = dsprites_dataloaders(
    path_dataset, val_ratio=0.1, num_workers=0, batch_size=256
)
x2c_extractor = get_dsprites_extractor_arch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _interv_npy(model_path, model_name, fold_0idx):
    npy = os.path.join(
        model_path,
        f"test_acc_y_random_group_level_True_use_prior_False_ints_{model_name}_fold_{fold_0idx}.npy"
    )
    return np.load(npy).tolist() if os.path.exists(npy) else None


def _split_results(model_path, model_name, split_idx):
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

    for fold_key in FOLDS:
        fold_1idx = int(fold_key.split("_")[1])
        fold_0idx = fold_1idx - 1

        r    = results[label][lam_c].get(fold_key, {})
        todo = {
            "task":  "task_acc"    not in r or RERUN_TASK,
            "interv":"interv"      not in r or RERUN_INTERV,
            "ois":   "ois_nis_cas" not in r or RERUN_OIS,
        }

        if todo["task"]:
            sr = _split_results(model_path, model_name, fold_0idx)
            if sr:
                r["task_acc"] = float(sr.get("test_acc_y", float("nan")))
                r["c_acc"]    = float(sr.get("test_acc_c", float("nan")))

        if todo["interv"]:
            curve = _interv_npy(model_path, model_name, fold_0idx)
            if curve is not None:
                r["interv"] = {"random": [curve]}

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
