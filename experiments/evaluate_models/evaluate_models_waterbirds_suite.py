"""
Waterbirds evaluation suite — CEM vs GC-CEM.

Metrics
-------
- Average test accuracy
- Worst-group test accuracy (min over 4 class×background groups)
- Per-group accuracy (4 values)
- Per-concept accuracy
- RTL_sum, RTL_norm, RCL_sum, RCL_norm (Ridge + MLP, global norm)
- CTL and ICL at concept probability level (KSG)
- Intervention curves (random policy)

RTL/RCL uses clean-split protocol: pool val+test → 5k/1k probe train/test.

Results saved to  results/results_waterbirds_suite.dict

Run from project root:
    python experiments/evaluate_models/evaluate_models_waterbirds_suite.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

master_folder = os.getcwd().replace("/experiments/evaluate_models", "")
sys.path.insert(0, master_folder)
master_folder = master_folder + "/"

import numpy as np
import joblib
import xai_concept_leakage.data.waterbirds_loader as wb_module
from xai_concept_leakage.metrics.leakage import compute_RTL_RCL, compute_RTL_RCL_mlp

from experiments.evaluate_models.eval_suite import (
    ALL_FOLDS, find_checkpoints, load_predictions,
    run_intervention_curve, compute_ctl, compute_icl,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRIAL_RUN     = False
RERUN_ALL     = False
RERUN_RTL_RCL = False
RERUN_CTL_ICL = False
RERUN_INTERV  = False

INTERVENTION_POLICIES = ["random"]
INTERVENTION_REPEATS  = 3

WB_FOLDER  = master_folder + "results/waterbirds/"
SAVE_PATH  = master_folder + "results/results_waterbirds_suite.dict"
LAM_C_LIST = ["lam_c0.1", "lam_c0.5", "lam_c1"]
FOLDS      = ["fold_1"] if TRIAL_RUN else ALL_FOLDS

# ---------------------------------------------------------------------------
# Clean-split for RTL/RCL  — pool val+test, never use training data
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
# Data
# ---------------------------------------------------------------------------
dataset_config = {
    "batch_size": 64, "num_workers": 0,
    "root_dir": master_folder + "data/",
    "train_augment": False, "weight_loss": False,
}
_, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = (
    wb_module.generate_data(config=dataset_config, seed=42, output_dataset_vars=True)
)
x2c_extractor = None
print(f"Waterbirds: n_concepts={n_concepts}, n_tasks={n_tasks}")
print(f"  test groups: {np.bincount(test_dl.dataset.groups, minlength=4)}")


def worst_group_accuracy(y_pred_classes, y_true, groups):
    per_group = {}
    for g in range(4):
        mask = groups == g
        if mask.sum() == 0:
            per_group[g] = float("nan")
        else:
            per_group[g] = float((y_pred_classes[mask] == y_true[mask]).mean())
    worst = min(v for v in per_group.values() if not np.isnan(v))
    return worst, per_group


# ---------------------------------------------------------------------------
# Metrics loop
# ---------------------------------------------------------------------------
results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

for LAM_C in LAM_C_LIST:
    cem_ckpts    = find_checkpoints(WB_FOLDER, "CEM_",   LAM_C, FOLDS)
    gc_cem_ckpts = find_checkpoints(WB_FOLDER, "GCCEM_", LAM_C, FOLDS)

    print(f"\n{'#'*60}")
    print(f"  LAM_C={LAM_C}")
    print(f"{'#'*60}")

    for label, fold_ckpts in [("cem", cem_ckpts), ("gc_cem", gc_cem_ckpts)]:
        if not fold_ckpts:
            print(f"  No {label.upper()} checkpoints for {LAM_C}, skipping.")
            continue
        results.setdefault(label, {}).setdefault(LAM_C, {})

        for fold, ckpt in fold_ckpts.items():
            r = results[label][LAM_C].get(fold, {})

            todo = {
                "task":    "task_acc" not in r or RERUN_ALL,
                "rtl_rcl": "rtl_rcl" not in r or RERUN_ALL or RERUN_RTL_RCL,
                "ctl_icl": "ctl_mean" not in r or RERUN_ALL or RERUN_CTL_ICL,
                "interv":  "interv"   not in r or RERUN_ALL or RERUN_INTERV,
            }

            if not any(todo.values()):
                print(f"\n  Skipping {label.upper()} {fold} ({LAM_C}) — all cached")
                continue

            print(f"\n{'='*60}")
            print(f"  {label.upper()} — {fold} — {LAM_C}")
            print(f"  Computing: {[k for k, v in todo.items() if v]}")
            print(f"{'='*60}")

            test_preds = load_predictions(ckpt, x2c_extractor, test_dl)
            val_preds  = load_predictions(ckpt, x2c_extractor, val_dl)
            N_CONCEPTS = test_preds["n_concepts"]
            EMB_SIZE   = test_preds["emb_size"]

            # --- task + worst-group accuracy ---
            if todo["task"]:
                y_pred_cls = test_preds["y_pred"].argmax(-1)
                y_true     = test_preds["y_true"]
                groups     = test_dl.dataset.groups
                r["task_acc"] = float((y_pred_cls == y_true).mean())
                r["c_acc"]    = float((test_preds["c_pred"] == test_preds["c_true"]).mean())
                worst, per_group = worst_group_accuracy(y_pred_cls, y_true, groups)
                r["worst_group_acc"] = worst
                r["per_group_acc"]   = per_group
            print(f"  task_acc={r['task_acc']:.4f}  worst_group={r['worst_group_acc']:.4f}  c_acc={r['c_acc']:.4f}")
            print(f"  per-group: " + "  ".join(f"g{g}={r['per_group_acc'][g]:.4f}" for g in range(4)))

            # --- RTL / RCL — pool val+test clean split ---
            if todo["rtl_rcl"]:
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

            # --- CTL / ICL at concept probability level ---
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
            results[label][LAM_C][fold] = r
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
    for lam_c in LAM_C_LIST:
        task_accs, worst_accs, rtl_norms, rcl_norms, ctl_means, icl_means = [], [], [], [], [], []
        for fold, r in results[label].get(lam_c, {}).items():
            if "task_acc" in r:
                task_accs.append(r["task_acc"])
                worst_accs.append(r["worst_group_acc"])
            if "rtl_rcl" in r:
                rtl_norms.append(r["rtl_rcl"]["ridge_global"]["RTL_norm"])
                rcl_norms.append(r["rtl_rcl"]["ridge_global"]["RCL_norm"])
            if "ctl_mean" in r:
                ctl_means.append(np.mean(r["ctl_mean"]))
                icl_means.append(r["icl_mean"])
        if task_accs:
            print(f"    [{lam_c}]  avg={np.mean(task_accs)*100:.2f}%  "
                  f"worst_group={np.mean(worst_accs)*100:.2f}±{np.std(worst_accs)*100:.2f}%"
                  + (f"  RTL_norm={np.mean(rtl_norms):.4f}  RCL_norm={np.mean(rcl_norms):.4f}"
                     if rtl_norms else "")
                  + (f"  CTL={np.mean(ctl_means):.4f}  ICL={np.mean(icl_means):.4f}"
                     if ctl_means else ""))
