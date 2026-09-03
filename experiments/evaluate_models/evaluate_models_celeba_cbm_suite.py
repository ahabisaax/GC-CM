"""
CelebA CBM evaluation suite.

Models
------
  joint   : SoftCBM lam_c0.1
  seq     : SeqCBM  lam_c1  (only variant trained)
  hard    : HardCBM (no lam_c)
  gc_cbm  : ACBM shared-critic lam_c0.1/0.5/1  — LAST checkpoint

Metrics
-------
  task_acc, c_acc, CTL, ICL (at c_prob level), intervention curves.
  No RTL/RCL — CBMs have no K×emb_size embedding space.
  Hard CBM: c_prob is binary, KSG CTL/ICL estimates will be unreliable (logged with warning).

Results saved to  results/results_celeba_cbm_suite.dict

Run from project root:
    python experiments/evaluate_models/evaluate_models_celeba_cbm_suite.py
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
from experiments.evaluate_model import predict_c_y
from experiments.evaluate_models.eval_suite import (
    ALL_FOLDS, run_intervention_curve, compute_ctl, compute_icl,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRIAL_RUN    = True   # fold_1 only; set False for full 5-fold run
RERUN_ALL    = False
RERUN_TASK   = False
RERUN_CTL_ICL = False
RERUN_INTERV  = False

INTERVENTION_POLICIES = ["random"]
INTERVENTION_REPEATS  = 3

CBM_FOLDER = master_folder + "results/celeba_cbm/"
SAVE_PATH  = master_folder + "results/results_celeba_cbm_suite.dict"
FOLDS      = ["fold_1"] if TRIAL_RUN else ALL_FOLDS

# (label, folder, checkpoint_prefix, lam_c_label, use_last_ckpt, is_hard)
MODELS = [
    dict(label="joint",  lam_c="lam_c0.1", last_ckpt=False, is_hard=False,
         folder=CBM_FOLDER + "SoftCBM_adam_lr5e-03_bs512_lam_c0.1/",
         prefix="SoftCBM_adam_lr5e-03_bs512_lam_c0.1"),
    dict(label="seq",    lam_c="none",     last_ckpt=False, is_hard=False,
         folder=CBM_FOLDER + "SeqCBM_adam_lr5e-03_bs512_lam_c1/",
         prefix="SeqCBM_adam_lr5e-03_bs512_lam_c1"),
    dict(label="hard",   lam_c="none",     last_ckpt=False, is_hard=True,
         folder=CBM_FOLDER + "HardCBM/",
         prefix="HardCBM"),
    dict(label="gc_cbm", lam_c="lam_c0.1", last_ckpt=True,  is_hard=False,
         folder=CBM_FOLDER + "ACBM_adam_lr5e-04_bs512_lam1_none_lam_c0.1_shared_critic/",
         prefix="ACBM_adam_lr5e-04_bs512_lam1_none_lam_c0.1_shared_critic"),
    dict(label="gc_cbm", lam_c="lam_c0.5", last_ckpt=True,  is_hard=False,
         folder=CBM_FOLDER + "ACBM_adam_lr5e-04_bs512_lam1_none_lam_c0.5_shared_critic/",
         prefix="ACBM_adam_lr5e-04_bs512_lam1_none_lam_c0.5_shared_critic"),
    dict(label="gc_cbm", lam_c="lam_c1",   last_ckpt=True,  is_hard=False,
         folder=CBM_FOLDER + "ACBM_adam_lr5e-04_bs512_lam1_none_lam_c1_shared_critic/",
         prefix="ACBM_adam_lr5e-04_bs512_lam1_none_lam_c1_shared_critic"),
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
    """Return path to best-checkpoint .pt file for a given fold."""
    fold_n = fold.replace("fold_", "")
    path = os.path.join(folder, f"{prefix}_fold_{fold_n}.pt")
    if os.path.exists(path):
        return path
    return None


def find_last_checkpoints(folder):
    """
    Return {fold: path} for last checkpoints, ordered by mtime (oldest=fold_1).
    Looks for last*.ckpt in folder/checkpoints/.
    """
    ckpt_dir = os.path.join(folder, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return {}
    files = sorted(glob.glob(os.path.join(ckpt_dir, "last*.ckpt")),
                   key=os.path.getmtime)
    fold_names = ALL_FOLDS[:len(files)]
    return {fold: path for fold, path in zip(fold_names, files)}


# ---------------------------------------------------------------------------
# CBM prediction (no embeddings — 4-element output)
# ---------------------------------------------------------------------------
def load_cbm_predictions(model_path, dl):
    """Run CBM inference; returns dict with c_prob, c_pred, c_true, y_pred, y_true."""
    out = predict_c_y(
        dl=dl,
        model_path=model_path,
        x2c_extractor=x2c_extractor,
        c_sem_out=False,
        soft_prob_out=True,
        vec_emb_out=False,
        torch_out=True,
    )
    # out = [c_prob, c_true, y_pred, y_true] as tensors
    c_prob_t, c_true_t, y_pred_t, y_true_t = out
    c_prob = c_prob_t.numpy().astype(np.float32)
    c_true = c_true_t.numpy().astype(np.float32)
    y_pred = y_pred_t.numpy()
    y_true = y_true_t.numpy().ravel().astype(np.int64)
    return {
        "c_prob": c_prob,
        "c_pred": (c_prob > 0.5).astype(float),
        "c_true": c_true,
        "y_pred": y_pred,
        "y_true": y_true,
        "n_concepts": int(c_true.shape[1]),
    }

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

for m in MODELS:
    label, lam_c, folder, prefix = m["label"], m["lam_c"], m["folder"], m["prefix"]
    use_last = m["last_ckpt"]
    is_hard  = m["is_hard"]

    if not os.path.isdir(folder):
        print(f"\n  Skipping {label} {lam_c} — folder not found: {folder}")
        continue

    # Build {fold: ckpt_path}
    if use_last:
        all_last = find_last_checkpoints(folder)
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
            "task":    "task_acc" not in r or RERUN_ALL or RERUN_TASK,
            "ctl_icl": "ctl_mean" not in r or RERUN_ALL or RERUN_CTL_ICL,
            "interv":  "interv"   not in r or RERUN_ALL or RERUN_INTERV,
        }

        if not any(todo.values()):
            print(f"\n  Skipping {label} {fold} ({lam_c}) — all cached")
            continue

        print(f"\n{'='*60}")
        print(f"  {label.upper()} — {fold} — {lam_c}  ckpt={os.path.basename(ckpt)}")
        print(f"  Computing: {[k for k, v in todo.items() if v]}")
        print(f"{'='*60}")

        if todo["task"] or todo["ctl_icl"]:
            print("  Loading predictions...")
            test_preds = load_cbm_predictions(ckpt, test_dl)
            K = test_preds["n_concepts"]

        # --- task acc + concept acc ---
        if todo["task"]:
            y_pred_cls = test_preds["y_pred"].argmax(-1)
            r["task_acc"] = float((y_pred_cls == test_preds["y_true"]).mean())
            r["c_acc"]    = float((test_preds["c_pred"] == test_preds["c_true"]).mean())
        print(f"  task_acc={r['task_acc']:.4f}  c_acc={r['c_acc']:.4f}")

        # --- CTL / ICL at concept probability level ---
        if todo["ctl_icl"]:
            c_prob = test_preds["c_prob"]
            if is_hard:
                print(f"  WARNING: Hard CBM — c_prob is binary, KSG CTL/ICL may be unreliable")
            print(f"  CTL (KSG, c_prob [{c_prob.shape}])...")
            ctl_mean, ctl_se = compute_ctl(c_prob, test_preds["y_true"], K)
            print(f"  ICL (KSG, c_prob [{c_prob.shape}])...")
            icl_mean, icl_se = compute_icl(c_prob, K)
            r["ctl_mean"] = ctl_mean.tolist() if hasattr(ctl_mean, "tolist") else list(ctl_mean)
            r["ctl_se"]   = ctl_se.tolist()   if hasattr(ctl_se,   "tolist") else list(ctl_se)
            r["icl_mean"] = icl_mean
            r["icl_se"]   = icl_se
            r["is_hard"]  = is_hard
        print(f"  CTL={np.mean(r['ctl_mean']):.4f}  ICL={r['icl_mean']:.4f}")

        # Save before intervention so task/CTL/ICL aren't lost on crash
        results[label][lam_c][fold] = r
        joblib.dump(results, SAVE_PATH)

        # --- Intervention curve ---
        if todo["interv"]:
            print(f"  Intervention curve ({INTERVENTION_POLICIES}, {INTERVENTION_REPEATS} repeats)...")
            r["interv"] = run_intervention_curve(
                ckpt, x2c_extractor, val_dl, val_dl, test_dl,
                policies=INTERVENTION_POLICIES,
                repeats=INTERVENTION_REPEATS,
            )
        if "interv" in r:
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
for label in ["joint", "seq", "hard", "gc_cbm"]:
    if label not in results:
        continue
    print(f"\n  {label.upper()}")
    for lam_c, folds_dict in results[label].items():
        task_accs, c_accs, ctl_means, icl_means = [], [], [], []
        interv_starts, interv_ends = [], []
        for fold, r in folds_dict.items():
            if "task_acc" in r:
                task_accs.append(r["task_acc"])
                c_accs.append(r["c_acc"])
            if "ctl_mean" in r:
                ctl_means.append(np.mean(r["ctl_mean"]))
                icl_means.append(r["icl_mean"])
            if "interv" in r:
                for policy, runs in r["interv"].items():
                    mc = np.mean(runs, axis=0)
                    interv_starts.append(mc[0])
                    interv_ends.append(mc[-1])
        if task_accs:
            line = (f"    [{lam_c}]  task={np.mean(task_accs)*100:.2f}%  "
                    f"c_acc={np.mean(c_accs)*100:.2f}%")
            if ctl_means:
                line += f"  CTL={np.mean(ctl_means):.4f}  ICL={np.mean(icl_means):.4f}"
            if interv_starts:
                line += (f"  interv: {np.mean(interv_starts):.4f}"
                         f" → {np.mean(interv_ends):.4f}")
            print(line)
