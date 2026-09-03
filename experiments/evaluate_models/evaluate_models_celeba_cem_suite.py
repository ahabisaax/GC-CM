"""
CelebA CEM evaluation suite: CEM vs GC-CEM.

Models
------
  cem      : CEM lam_c0.1 / lam_c0.5           — BEST checkpoint
  gc_cem   : CRCEM shared-critic lam_c0.1 / 0.5 — LAST checkpoint

Metrics
-------
  task_acc, c_acc
  CTL, ICL (KSG, at c_prob level)
  Intervention curves (random policy)
  RTL_sum, RTL_norm, RCL_sum, RCL_norm (ridge_global + mlp_global, global_norm=True)

Results saved to  results/results_celeba_cem_suite.dict

Run from project root:
    python experiments/evaluate_models/evaluate_models_celeba_cem_suite.py
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")

master_folder = os.getcwd().replace("/experiments/evaluate_models", "")
sys.path.insert(0, master_folder)
master_folder = master_folder + "/"

import numpy as np
import joblib
import torch
import torch.utils.data as tud

import xai_concept_leakage.data.celeba_loader as celeba_data_module
from experiments.evaluate_models.eval_suite import (
    ALL_FOLDS,
    load_predictions,
    run_intervention_curve,
    compute_ctl,
    compute_icl,
)
from xai_concept_leakage.metrics.leakage import compute_RTL_RCL, compute_RTL_RCL_mlp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRIAL_RUN         = True   # fold_1 only; set False for full 5-fold run
RERUN_ALL         = False
RERUN_TASK        = False
RERUN_CTL_ICL     = False
RERUN_INTERV      = False
RERUN_RTL_RCL     = False

INTERVENTION_POLICIES = ["random"]
INTERVENTION_REPEATS  = 3

CEM_FOLDER = master_folder + "results/celeba_cem_39c/"
SAVE_PATH  = master_folder + "results/results_celeba_cem_suite.dict"
FOLDS      = ["fold_1"] if TRIAL_RUN else ALL_FOLDS

MODELS = [
    dict(label="cem",    lam_c="lam_c0.1", last_ckpt=False,
         folder=CEM_FOLDER + "CEM_adam_lr1e-03_bs512_lam_c0.1/",
         prefix="CEM_adam_lr1e-03_bs512_lam_c0.1"),
    dict(label="cem",    lam_c="lam_c0.5", last_ckpt=False,
         folder=CEM_FOLDER + "CEM_adam_lr1e-03_bs512_lam_c0.5/",
         prefix="CEM_adam_lr1e-03_bs512_lam_c0.5"),
    dict(label="cem",    lam_c="lam_c1",   last_ckpt=False,
         folder=CEM_FOLDER + "CEM_adam_lr1e-03_bs512_lam_c1.0/",
         prefix="CEM_adam_lr1e-03_bs512_lam_c1.0"),
    dict(label="gc_cem", lam_c="lam_c0.1", last_ckpt=True,
         folder=CEM_FOLDER + "CRCEM_adam_lr5e-04_bs512_lam1_none_lam_c0.1_shared_critic/",
         prefix="CRCEM_adam_lr5e-04_bs512_lam1_none_lam_c0.1_shared_critic"),
    dict(label="gc_cem", lam_c="lam_c0.5", last_ckpt=True,
         folder=CEM_FOLDER + "CRCEM_adam_lr5e-04_bs512_lam1_none_lam_c0.5_shared_critic/",
         prefix="CRCEM_adam_lr5e-04_bs512_lam1_none_lam_c0.5_shared_critic"),
    dict(label="gc_cem", lam_c="lam_c1",   last_ckpt=True,
         folder=CEM_FOLDER + "CRCEM_adam_lr5e-04_bs512_lam1_none_lam_c1.0_shared_critic/",
         prefix="CRCEM_adam_lr5e-04_bs512_lam1_none_lam_c1.0_shared_critic"),
]

# ---------------------------------------------------------------------------
# CelebA DataLoader  (x,(y,c)) → (x,y,c)
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
_, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = (
    celeba_data_module.generate_data(
        config=dataset_config, seed=42,
        output_dataset_vars=True,
        root_dir=dataset_config["root_dir"],
    )
)
val_dl  = wrap_celeba_dl(val_dl)
test_dl = wrap_celeba_dl(test_dl)
x2c_extractor = None
print(f"CelebA: n_concepts={n_concepts}, n_tasks={n_tasks}")

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def find_best_checkpoint(folder, prefix, fold):
    fold_n = fold.replace("fold_", "")
    path = os.path.join(folder, f"{prefix}_fold_{fold_n}.pt")
    return path if os.path.exists(path) else None


def find_last_checkpoints(folder):
    """Return {fold: path} ordered by mtime (oldest=fold_1)."""
    ckpt_dir = os.path.join(folder, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return {}
    files = sorted(glob.glob(os.path.join(ckpt_dir, "last*.ckpt")),
                   key=os.path.getmtime)
    fold_names = ALL_FOLDS[:len(files)]
    return {fold: path for fold, path in zip(fold_names, files)}


# ---------------------------------------------------------------------------
# RTL / RCL: pool val+test and split into probe-train / probe-test
# ---------------------------------------------------------------------------
def pool_and_split(preds_val, preds_test, n_tr=5000, n_te=1000, seed=42):
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
# Evaluation loop
# ---------------------------------------------------------------------------
results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

for m in MODELS:
    label, lam_c = m["label"], m["lam_c"]
    folder, prefix, use_last = m["folder"], m["prefix"], m["last_ckpt"]

    if not os.path.isdir(folder):
        print(f"\n  Skipping {label} {lam_c} — folder not found: {folder}")
        continue

    if use_last:
        all_last  = find_last_checkpoints(folder)
        fold_ckpts = {f: all_last[f] for f in FOLDS if f in all_last}
    else:
        fold_ckpts = {}
        for fold in FOLDS:
            path = find_best_checkpoint(folder, prefix, fold)
            if path:
                fold_ckpts[fold] = path

    if not fold_ckpts:
        print(f"\n  Skipping {label} {lam_c} — no checkpoints found in {folder}")
        continue

    results.setdefault(label, {}).setdefault(lam_c, {})

    print(f"\n{'#'*60}")
    print(f"  {label.upper()} — {lam_c}  {'[LAST CKPT]' if use_last else '[BEST CKPT]'}")
    print(f"{'#'*60}")

    for fold, ckpt in fold_ckpts.items():
        r = results[label][lam_c].get(fold, {})

        todo = {
            "task":    "task_acc"  not in r or RERUN_ALL or RERUN_TASK,
            "rtl_rcl": "rtl_rcl"   not in r or RERUN_ALL or RERUN_RTL_RCL,
            "ctl_icl": "ctl_mean"  not in r or RERUN_ALL or RERUN_CTL_ICL,
            "interv":  "interv"    not in r or RERUN_ALL or RERUN_INTERV,
        }

        if not any(todo.values()):
            print(f"\n  Skipping {label} {fold} ({lam_c}) — all cached")
            continue

        print(f"\n{'='*60}")
        print(f"  {label.upper()} — {fold} — {lam_c}  ckpt={os.path.basename(ckpt)}")
        print(f"  Computing: {[k for k, v in todo.items() if v]}")
        print(f"{'='*60}")

        need_preds = todo["task"] or todo["rtl_rcl"] or todo["ctl_icl"]
        if need_preds:
            print("  Loading test predictions...")
            test_preds = load_predictions(ckpt, x2c_extractor, test_dl)
            N_CONCEPTS = test_preds["n_concepts"]
            EMB_SIZE   = test_preds["emb_size"]

        # --- task + concept acc ---
        if todo["task"]:
            y_pred_cls   = test_preds["y_pred"].argmax(-1)
            r["task_acc"] = float((y_pred_cls == test_preds["y_true"]).mean())
            r["c_acc"]    = float((test_preds["c_pred"] == test_preds["c_true"]).mean())
        print(f"  task_acc={r['task_acc']:.4f}  c_acc={r['c_acc']:.4f}")

        # --- RTL / RCL (global norm, ridge + MLP) ---
        if todo["rtl_rcl"]:
            print("  Loading val predictions for RTL/RCL probe split...")
            val_preds = load_predictions(ckpt, x2c_extractor, val_dl)
            c_mix_tr, c_true_tr, y_tr, c_mix_te, c_true_te, y_te = pool_and_split(
                val_preds, test_preds,
            )
            rng = np.random.RandomState(42)
            mlp_idx = rng.permutation(len(y_tr))[:2000]
            ridge_args = (c_mix_tr, c_mix_te, c_true_tr, c_true_te, y_tr, y_te)
            mlp_args   = (c_mix_tr[mlp_idx], c_mix_te,
                          c_true_tr[mlp_idx], c_true_te,
                          y_tr[mlp_idx], y_te)
            print(f"  RTL/RCL Ridge (global norm, {N_CONCEPTS}×{EMB_SIZE}-dim, n_tr={len(y_tr)})...")
            ridge = compute_RTL_RCL(*ridge_args, global_norm=True)
            print(f"  RTL/RCL MLP   (global norm, n_tr={mlp_idx.shape[0]})...")
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

        # --- CTL / ICL at c_prob level ---
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

        # --- Intervention curve ---
        if todo["interv"]:
            print(f"  Intervention curve ({INTERVENTION_POLICIES}, {INTERVENTION_REPEATS} repeats)...")
            r["interv"] = run_intervention_curve(
                ckpt, x2c_extractor, val_dl, val_dl, test_dl,
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
for label in ["cem", "gc_cem"]:
    if label not in results:
        continue
    print(f"\n  {label.upper()}")
    for lam_c, folds_dict in results[label].items():
        task_accs, c_accs, ctl_means, icl_means = [], [], [], []
        rtl_norms_r, rcl_norms_r = [], []
        rtl_norms_m, rcl_norms_m = [], []
        interv_starts, interv_ends = [], []
        for fold, r in folds_dict.items():
            if "task_acc" in r:
                task_accs.append(r["task_acc"])
                c_accs.append(r["c_acc"])
            if "ctl_mean" in r:
                ctl_means.append(np.mean(r["ctl_mean"]))
                icl_means.append(r["icl_mean"])
            if "rtl_rcl" in r:
                rtl_norms_r.append(r["rtl_rcl"]["ridge_global"]["RTL_norm"])
                rcl_norms_r.append(r["rtl_rcl"]["ridge_global"]["RCL_norm"])
                rtl_norms_m.append(r["rtl_rcl"]["mlp_global"]["RTL_norm"])
                rcl_norms_m.append(r["rtl_rcl"]["mlp_global"]["RCL_norm"])
            if "interv" in r:
                for policy, runs in r["interv"].items():
                    curve = np.mean(runs, axis=0)
                    interv_starts.append(curve[0])
                    interv_ends.append(curve[-1])
        print(f"    {lam_c}  ({len(task_accs)} folds)")
        if task_accs:
            print(f"      task={np.mean(task_accs):.4f}  c_acc={np.mean(c_accs):.4f}")
        if ctl_means:
            print(f"      CTL={np.mean(ctl_means):.4f}  ICL={np.mean(icl_means):.4f}")
        if rtl_norms_r:
            print(f"      Ridge: RTL_norm={np.mean(rtl_norms_r):.4f}  RCL_norm={np.mean(rcl_norms_r):.4f}")
            print(f"      MLP:   RTL_norm={np.mean(rtl_norms_m):.4f}  RCL_norm={np.mean(rcl_norms_m):.4f}")
        if interv_starts:
            print(f"      Interv: {np.mean(interv_starts):.4f} → {np.mean(interv_ends):.4f}")
