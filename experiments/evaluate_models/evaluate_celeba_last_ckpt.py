"""
CelebA eval using last-epoch checkpoints (_last.pt files).

Computes for CEM and GC-CEM at all lam_c values:
  - task accuracy, concept accuracy
  - RTL_sum, RTL_norm, RCL_sum, RCL_norm (Ridge + MLP, global norm)
  - CTL and ICL at concept probability level (KSG)
  - Intervention curves (random policy)

RTL/RCL uses the same clean-split protocol as other datasets:
  pool val+test → shuffle → 30k probe-train / 9k probe-test.

Results saved to:  results/results_celeba_suite_last_ckpt.dict
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
    ALL_FOLDS, find_checkpoints, load_predictions,
    run_intervention_curve, compute_ctl, compute_icl,
)

# ---------------------------------------------------------------------------
# DataLoader wrapper  (x,(y,c)) → (x,y,c)
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
    return tud.DataLoader(
        CelebADatasetWrapper(dl.dataset),
        batch_size=dl.batch_size, shuffle=False,
        num_workers=dl.num_workers, pin_memory=dl.pin_memory,
    )

# ---------------------------------------------------------------------------
# Clean-split for RTL/RCL  — matches cache_embeddings_all.py exactly
# ---------------------------------------------------------------------------

def pool_and_split(preds_val, preds_test, n_tr=30000, n_te=9000, seed=42):
    """Pool val+test predictions, shuffle, split into probe train/test."""
    c_mix  = np.concatenate([preds_val["c_mix"],  preds_test["c_mix"]],  axis=0)
    c_true = np.concatenate([preds_val["c_true"], preds_test["c_true"]], axis=0)
    y      = np.concatenate([preds_val["y_true"], preds_test["y_true"]], axis=0)
    N = len(y)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    n_tr_actual = min(n_tr, N - n_te)
    tr_idx = idx[:n_tr_actual]
    te_idx = idx[n_tr_actual:n_tr_actual + n_te]
    print(f"    pool: total={N} → probe_train={n_tr_actual}  probe_test={n_te}")
    return (
        c_mix[tr_idx].astype(np.float32),  c_true[tr_idx].astype(np.float32),
        y[tr_idx].astype(np.int64),
        c_mix[te_idx].astype(np.float32),  c_true[te_idx].astype(np.float32),
        y[te_idx].astype(np.int64),
    )

# ---------------------------------------------------------------------------
# Last-checkpoint finder
# ---------------------------------------------------------------------------

def find_last_checkpoints(folder, prefix, lam_c, folds):
    """
    Find <model_dir>/<name>_fold_X_last.pt files.
    Falls back to Lightning last*.ckpt (ordered by mtime = fold order).
    """
    model_dir = None
    for d in sorted(os.listdir(folder)):
        if not os.path.isdir(os.path.join(folder, d)) or d == "auto":
            continue
        if d.startswith(prefix) and lam_c in d:
            model_dir = d
            break
    if not model_dir:
        print(f"  No model dir matching '{prefix}' + '{lam_c}' in {folder}")
        return {}

    model_path = os.path.join(folder, model_dir)
    checkpoints = {}
    for fold in folds:
        fold_num = fold.split("_")[-1]
        last_pt = os.path.join(model_path, f"{model_dir}_fold_{fold_num}_last.pt")
        if os.path.exists(last_pt):
            checkpoints[fold] = last_pt
            continue
        # Fall back to Lightning last*.ckpt ordered by mtime
        ckpt_dir = os.path.join(model_path, "checkpoints")
        if os.path.isdir(ckpt_dir):
            last_files = sorted(
                [f for f in os.listdir(ckpt_dir) if f.startswith("last")],
                key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)),
            )
            idx = int(fold_num) - 1
            if idx < len(last_files):
                checkpoints[fold] = os.path.join(ckpt_dir, last_files[idx])
                continue
        print(f"  WARNING: no last checkpoint for {fold} in {model_dir}")
    return checkpoints

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RERUN_ALL   = False   # set True to recompute everything
LAM_C_LIST  = ["lam_c0.1", "lam_c0.5", "lam_c1"]
_FOLDER     = master_folder + "results/celeba_cem_39c/"
MODEL_SPECS = [
    ("cem",   "CEM_",   _FOLDER),
    ("crcem", "CRCEM_", _FOLDER),
]
SAVE_PATH = master_folder + "results/results_celeba_suite_last_ckpt.dict"
INTERVENTION_POLICIES = ["random"]
INTERVENTION_REPEATS  = 3

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

dataset_config = {
    "dataset": "celeba", "num_workers": 0, "batch_size": 256,
    "root_dir": master_folder + "data/",
    "label_attr_idx": 20, "num_concepts": 39, "num_classes": 2,
    "image_size": 64, "weight_loss": False, "train_augment": False,
}
_, val_dl, test_dl, _, (n_concepts, n_tasks, _) = celeba_data_module.generate_data(
    config=dataset_config, seed=42, output_dataset_vars=True,
    root_dir=dataset_config["root_dir"],
)
val_dl  = wrap_celeba_dl(val_dl)
test_dl = wrap_celeba_dl(test_dl)
print(f"CelebA: n_concepts={n_concepts}  val={len(val_dl.dataset)}  test={len(test_dl.dataset)}")

# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

for label, prefix, folder in MODEL_SPECS:
    results.setdefault(label, {})

    for lam_c in LAM_C_LIST:
        fold_ckpts = find_last_checkpoints(folder, prefix, lam_c, ALL_FOLDS)
        if not fold_ckpts:
            print(f"\n  Skipping {label.upper()} {lam_c} — no last checkpoints found")
            continue

        results[label].setdefault(lam_c, {})
        print(f"\n{'#'*60}")
        print(f"  {label.upper()} — {lam_c} [last checkpoint]")
        print(f"{'#'*60}")

        for fold, ckpt in fold_ckpts.items():
            r = results[label][lam_c].get(fold, {})

            todo = {
                "task":    "task_acc" not in r or RERUN_ALL,
                "rtl_rcl": "rtl_rcl" not in r or RERUN_ALL,
                "ctl_icl": "ctl_mean" not in r or RERUN_ALL,
                "interv":  "interv"   not in r or RERUN_ALL,
            }
            if not any(todo.values()):
                print(f"\n  Skipping {fold} — all cached")
                continue

            print(f"\n{'='*60}")
            print(f"  {label.upper()} — {fold} — {lam_c}")
            print(f"  ckpt: {os.path.relpath(ckpt, master_folder)}")
            print(f"  Computing: {[k for k, v in todo.items() if v]}")
            print(f"{'='*60}")

            test_preds = load_predictions(ckpt, None, test_dl)
            val_preds  = load_predictions(ckpt, None, val_dl)
            N_CONCEPTS = test_preds["n_concepts"]
            EMB_SIZE   = test_preds["emb_size"]

            # task + concept accuracy
            if todo["task"]:
                r["task_acc"] = float((test_preds["y_pred"].argmax(-1) == test_preds["y_true"]).mean())
                r["c_acc"]    = float((test_preds["c_pred"] == test_preds["c_true"]).mean())
            print(f"  task_acc={r['task_acc']:.4f}  c_acc={r['c_acc']:.4f}")

            # RTL / RCL — pool val+test, clean split
            if todo["rtl_rcl"]:
                c_mix_tr, c_true_tr, y_tr, c_mix_te, c_true_te, y_te = pool_and_split(
                    val_preds, test_preds,
                )
                rng = np.random.RandomState(42)
                mlp_idx = rng.permutation(len(y_tr))[:5000]
                ridge_args = (c_mix_tr, c_mix_te, c_true_tr, c_true_te, y_tr, y_te)
                mlp_args   = (c_mix_tr[mlp_idx], c_mix_te,
                              c_true_tr[mlp_idx], c_true_te,
                              y_tr[mlp_idx], y_te)
                print(f"  RTL/RCL Ridge (global norm, {N_CONCEPTS}×{EMB_SIZE}-dim, n_tr={len(y_tr)})...")
                ridge = compute_RTL_RCL(*ridge_args, global_norm=True)
                print(f"  RTL/RCL MLP   (global norm, n_tr=5000)...")
                mlp   = compute_RTL_RCL_mlp(*mlp_args, global_norm=True, hidden=(64,), max_iter=200)
                r["rtl_rcl"] = {"ridge_global": ridge, "mlp_global": mlp}
            rr = r["rtl_rcl"]
            print(f"  Ridge: RTL_sum={rr['ridge_global']['RTL_sum']:.4f}  "
                  f"RTL_norm={rr['ridge_global']['RTL_norm']:.4f}  "
                  f"RCL_sum={rr['ridge_global']['RCL_sum']:.4f}  "
                  f"RCL_norm={rr['ridge_global']['RCL_norm']:.4f}")
            print(f"  MLP:   RTL_sum={rr['mlp_global']['RTL_sum']:.4f}  "
                  f"RTL_norm={rr['mlp_global']['RTL_norm']:.4f}  "
                  f"RCL_sum={rr['mlp_global']['RCL_sum']:.4f}  "
                  f"RCL_norm={rr['mlp_global']['RCL_norm']:.4f}")

            # CTL / ICL
            if todo["ctl_icl"]:
                c_prob = test_preds["c_prob"]
                print(f"  CTL (KSG, c_prob [{c_prob.shape}])...")
                ctl_mean, ctl_se = compute_ctl(c_prob, test_preds["y_true"], N_CONCEPTS)
                print(f"  ICL (KSG, c_prob [{c_prob.shape}])...")
                icl_mean, icl_se = compute_icl(c_prob, N_CONCEPTS)
                r["ctl_mean"] = ctl_mean.tolist() if hasattr(ctl_mean, "tolist") else list(ctl_mean)
                r["ctl_se"]   = ctl_se.tolist()   if hasattr(ctl_se,   "tolist") else list(ctl_se)
                r["icl_mean"] = icl_mean
                r["icl_se"]   = icl_se
            print(f"  CTL={np.mean(r['ctl_mean']):.4f}  ICL={r['icl_mean']:.4f}")

            # Interventions
            if todo["interv"]:
                print(f"  Intervention curve ({INTERVENTION_POLICIES}, {INTERVENTION_REPEATS} repeats)...")
                r["interv"] = run_intervention_curve(
                    ckpt, None, val_dl, val_dl, test_dl,
                    policies=INTERVENTION_POLICIES,
                    repeats=INTERVENTION_REPEATS,
                )
            for policy, runs in r["interv"].items():
                mean_curve = np.mean(runs, axis=0)
                print(f"  Intervention [{policy}]: 0={mean_curve[0]:.4f} → all={mean_curve[-1]:.4f}")

            r["_complete"] = True
            results[label][lam_c][fold] = r
            joblib.dump(results, SAVE_PATH)
            print(f"  Saved → {SAVE_PATH}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
joblib.dump(results, SAVE_PATH)
print(f"\nResults saved → {SAVE_PATH}")
print("\n=== Summary ===")
for label, _, _ in MODEL_SPECS:
    if label not in results:
        continue
    print(f"\n  {label.upper()}")
    for lam_c in LAM_C_LIST:
        task_accs, rtl_norms, rcl_norms, ctl_means, icl_means = [], [], [], [], []
        for fold, r in results[label].get(lam_c, {}).items():
            if "task_acc" in r:  task_accs.append(r["task_acc"])
            if "rtl_rcl"  in r:
                rtl_norms.append(r["rtl_rcl"]["ridge_global"]["RTL_norm"])
                rcl_norms.append(r["rtl_rcl"]["ridge_global"]["RCL_norm"])
            if "ctl_mean" in r:
                ctl_means.append(np.mean(r["ctl_mean"]))
                icl_means.append(r["icl_mean"])
        if task_accs:
            print(f"    [{lam_c}]  task={np.mean(task_accs)*100:.2f}±{np.std(task_accs)*100:.2f}%"
                  + (f"  RTL_norm={np.mean(rtl_norms):.4f}±{np.std(rtl_norms):.4f}"
                     f"  RCL_norm={np.mean(rcl_norms):.4f}" if rtl_norms else "")
                  + (f"  CTL={np.mean(ctl_means):.4f}  ICL={np.mean(icl_means):.4f}"
                     if ctl_means else ""))
