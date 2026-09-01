"""
CEM/CRCEM evaluation suite for CelebA (39 concepts, 2 classes: Male/not-Male).

Metrics
-------
- Task accuracy, per-concept accuracy
- RTL / RCL (Ridge + MLP, global norm) — canonical leakage metrics
- CTL (KSG MI, concept probabilities vs task label)
- ICL (KSG MI, pairwise concept probabilities)
- Intervention curve (random policy, group-level)

Folder layout
-------------
CEM   lam_c0.1 → results/celeba_cem_39c/
CEM   lam_c1   → results/celeba_cem_39c_lam1_0/
CRCEM lam_c0.1 → results/celeba_cem_39c/
CRCEM lam_c0.5 → results/celeba_crcem_39c_lam0_5/
CRCEM lam_c1   → results/celeba_cem_39c_lam1_0/

Results saved to  results/results_celeba_suite.dict
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

master_folder = os.getcwd().replace("/experiments/evaluate_models", "")
sys.path.insert(0, master_folder)
master_folder = master_folder + "/"

import numpy as np
import joblib
import torch.utils.data as tud

import xai_concept_leakage.data.celeba_loader as celeba_data_module
from xai_concept_leakage.metrics.leakage import compute_RTL_RCL, compute_RTL_RCL_mlp

from experiments.evaluate_models.eval_suite import (
    ALL_FOLDS, find_checkpoints,
    load_predictions,
    run_intervention_curve,
    compute_ctl, compute_icl,
)

# ---------------------------------------------------------------------------
# CelebA DataLoader wrapper  (x,(y,c)) → (x,y,c)
# ---------------------------------------------------------------------------

class CelebADatasetWrapper(tud.Dataset):
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        x, (y, c) = self.ds[idx]
        return x, y, c


def wrap_celeba_dl(dl):
    wrapped_ds = CelebADatasetWrapper(dl.dataset)
    return tud.DataLoader(
        wrapped_ds,
        batch_size=dl.batch_size,
        shuffle=isinstance(dl.sampler, tud.RandomSampler),
        num_workers=dl.num_workers,
        pin_memory=dl.pin_memory,
    )

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRIAL_RUN      = False
RERUN_TASK     = False
RERUN_RTL_RCL  = True   # recompute with corrected 5k subsample
RERUN_CTL_ICL  = False
RERUN_INTERV   = False

INTERVENTION_POLICIES = ["random"]
INTERVENTION_REPEATS  = 3
MAX_PROBE_SAMPLES     = 5000    # subsample train for RTL/RCL — matches dSprites cache_embeddings split

# All CelebA CEM/CRCEM models live in one folder
_CELEBA_FOLDER = master_folder + "results/celeba_cem_39c/"
MODEL_FOLDERS = {
    "cem":   {"lam_c0.1": _CELEBA_FOLDER, "lam_c0.5": _CELEBA_FOLDER, "lam_c1": _CELEBA_FOLDER},
    "crcem": {"lam_c0.1": _CELEBA_FOLDER, "lam_c0.5": _CELEBA_FOLDER, "lam_c1": _CELEBA_FOLDER},
}

FOLDS     = ["fold_1"] if TRIAL_RUN else ALL_FOLDS
SAVE_PATH = master_folder + "results/results_celeba_suite.dict"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
dataset_config = {
    "dataset": "celeba",
    "num_workers": 0,
    "batch_size": 256,
    "root_dir": master_folder + "data/",
    "label_attr_idx": 20,
    "num_concepts": 39,
    "num_classes": 2,
    "image_size": 64,
    "weight_loss": False,
    "train_augment": False,
}
train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = (
    celeba_data_module.generate_data(
        config=dataset_config, seed=42,
        output_dataset_vars=True,
        root_dir=dataset_config["root_dir"],
    )
)
train_dl = wrap_celeba_dl(train_dl)
val_dl   = wrap_celeba_dl(val_dl)
test_dl  = wrap_celeba_dl(test_dl)
x2c_extractor = None
print(f"CelebA: n_concepts={n_concepts}, n_tasks={n_tasks}")

# ---------------------------------------------------------------------------
# Metrics loop
# ---------------------------------------------------------------------------
results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

for label, lam_folders in MODEL_FOLDERS.items():
    prefix = "CEM_" if label == "cem" else "CRCEM_"
    results.setdefault(label, {})

    for LAM_C, folder in lam_folders.items():
        if not os.path.isdir(folder):
            print(f"\n  Skipping {label.upper()} {LAM_C} — folder not found: {folder}")
            continue

        fold_ckpts = find_checkpoints(folder, prefix, LAM_C, FOLDS)
        if not fold_ckpts:
            print(f"  No {label.upper()} checkpoints for {LAM_C}, skipping.")
            continue

        results[label].setdefault(LAM_C, {})

        print(f"\n{'#'*60}")
        print(f"  {label.upper()} — {LAM_C} — {os.path.basename(folder.rstrip('/'))}")
        print(f"{'#'*60}")

        for fold, ckpt in fold_ckpts.items():
            r = results[label][LAM_C].get(fold, {})

            todo = {
                "task":    "task_acc"  not in r or RERUN_TASK,
                "rtl_rcl": "rtl_rcl"  not in r or RERUN_RTL_RCL,
                "ctl_icl": "ctl_mean" not in r or RERUN_CTL_ICL,
                "interv":  "interv"   not in r or RERUN_INTERV,
            }

            if not any(todo.values()):
                print(f"\n  Skipping {label.upper()} {fold} ({LAM_C}) — all cached")
                continue

            print(f"\n{'='*60}")
            print(f"  {label.upper()} — {fold} — {LAM_C}")
            print(f"  Computing: {[k for k, v in todo.items() if v]}")
            print(f"{'='*60}")

            print("  Loading predictions...")
            test_preds  = load_predictions(ckpt, x2c_extractor, test_dl)
            train_preds = load_predictions(ckpt, x2c_extractor, train_dl)

            N_CONCEPTS = test_preds["n_concepts"]
            EMB_SIZE   = test_preds["emb_size"]

            # --- task + concept accuracy ---
            if todo["task"]:
                r["task_acc"] = float((test_preds["y_pred"].argmax(-1) == test_preds["y_true"]).mean())
                r["c_acc"]    = float((test_preds["c_pred"] == test_preds["c_true"]).mean())
            print(f"  task_acc={r['task_acc']:.4f}  c_acc={r['c_acc']:.4f}")

            # --- RTL / RCL (Ridge + MLP, global norm) ---
            if todo["rtl_rcl"]:
                # Subsample train to MAX_PROBE_SAMPLES — full CelebA train (162k) is
                # too slow for sklearn MLP; consistent with cache_embeddings_all.py split sizes.
                rng = np.random.RandomState(42)
                n_tr = len(train_preds["y_true"])
                idx  = rng.permutation(n_tr)[:MAX_PROBE_SAMPLES]
                args = (
                    train_preds["c_mix"][idx], test_preds["c_mix"],
                    train_preds["c_true"][idx], test_preds["c_true"],
                    train_preds["y_true"][idx], test_preds["y_true"],
                )
                print(f"  Computing RTL/RCL Ridge (global norm, {N_CONCEPTS}×{EMB_SIZE}-dim)...")
                ridge = compute_RTL_RCL(*args, global_norm=True)
                print(f"  Computing RTL/RCL MLP (global norm)...")
                mlp   = compute_RTL_RCL_mlp(*args, global_norm=True, hidden=(64,), max_iter=200)
                r["rtl_rcl"] = {"ridge_global": ridge, "mlp_global": mlp}
            rr = r["rtl_rcl"]
            print(f"  Ridge: RTL_sum={rr['ridge_global']['RTL_sum']:.4f}  RTL_norm={rr['ridge_global']['RTL_norm']:.4f}"
                  f"  RCL_sum={rr['ridge_global']['RCL_sum']:.4f}  RCL_norm={rr['ridge_global']['RCL_norm']:.4f}")
            print(f"  MLP:   RTL_sum={rr['mlp_global']['RTL_sum']:.4f}  RTL_norm={rr['mlp_global']['RTL_norm']:.4f}"
                  f"  RCL_sum={rr['mlp_global']['RCL_sum']:.4f}  RCL_norm={rr['mlp_global']['RCL_norm']:.4f}")

            # --- CTL / ICL at concept probability level ---
            if todo["ctl_icl"]:
                # c_prob is [N, K] — pass directly as c_mix_flat (m=1 per concept)
                c_prob_test  = test_preds["c_prob"]
                c_prob_train = train_preds["c_prob"]
                print(f"  Computing CTL (KSG, c_prob [{c_prob_test.shape}])...")
                ctl_mean, ctl_se = compute_ctl(
                    c_prob_test, test_preds["y_true"], N_CONCEPTS,
                )
                print(f"  Computing ICL (KSG, c_prob [{c_prob_test.shape}])...")
                icl_mean, icl_se = compute_icl(c_prob_test, N_CONCEPTS)
                r["ctl_mean"] = ctl_mean.tolist() if hasattr(ctl_mean, "tolist") else list(ctl_mean)
                r["ctl_se"]   = ctl_se.tolist()   if hasattr(ctl_se,   "tolist") else list(ctl_se)
                r["icl_mean"] = icl_mean
                r["icl_se"]   = icl_se
            print(f"  CTL (mean over concepts)={np.mean(r['ctl_mean']):.4f}")
            print(f"  ICL={r['icl_mean']:.4f} ± {r['icl_se']:.4f}")

            # --- Intervention curve ---
            if todo["interv"]:
                print(f"  Computing intervention curve ({INTERVENTION_POLICIES}, {INTERVENTION_REPEATS} repeats)...")
                r["interv"] = run_intervention_curve(
                    ckpt, x2c_extractor, train_dl, val_dl, test_dl,
                    policies=INTERVENTION_POLICIES,
                    repeats=INTERVENTION_REPEATS,
                )
            for policy, runs in r["interv"].items():
                mean_curve = np.mean(runs, axis=0)
                print(f"  Intervention [{policy}]: 0={mean_curve[0]:.4f} → all={mean_curve[-1]:.4f}")

            r["_complete"] = True
            results[label][LAM_C][fold] = r
            joblib.dump(results, SAVE_PATH)
            print(f"  Checkpoint saved → {SAVE_PATH}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
joblib.dump(results, SAVE_PATH)
print(f"\nResults saved → {SAVE_PATH}")

print("\n=== Summary ===")
for label in ["cem", "crcem"]:
    if label not in results:
        continue
    print(f"\n  {label.upper()}")
    for lam_c in sorted(results[label]):
        task_accs, rtl_norms, rcl_norms, ctl_means, icl_means = [], [], [], [], []
        for fold, r in results[label][lam_c].items():
            if "task_acc" in r:
                task_accs.append(r["task_acc"])
            if "rtl_rcl" in r:
                rtl_norms.append(r["rtl_rcl"]["ridge_global"]["RTL_norm"])
                rcl_norms.append(r["rtl_rcl"]["ridge_global"]["RCL_norm"])
            if "ctl_mean" in r:
                ctl_means.append(np.mean(r["ctl_mean"]))
                icl_means.append(r["icl_mean"])
        if task_accs:
            print(f"    [{lam_c}]  task={np.mean(task_accs)*100:.2f}±{np.std(task_accs)*100:.2f}%"
                  + (f"  RTL_norm={np.mean(rtl_norms):.4f}  RCL_norm={np.mean(rcl_norms):.4f}" if rtl_norms else "")
                  + (f"  CTL={np.mean(ctl_means):.4f}  ICL={np.mean(icl_means):.4f}" if ctl_means else ""))
