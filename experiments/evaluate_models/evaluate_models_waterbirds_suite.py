"""
Waterbirds evaluation suite — CEM vs GC-CEM.

Metrics
-------
- Average test accuracy
- Worst-group test accuracy (min over 4 class×background groups)
- Per-group accuracy (4 values)
- Per-concept accuracy
- CVL (concept variance leakage, global)
- ICVL (inter-concept variance leakage)
- Task leakage probe (y|c vs y|[c, ĉ])
- Embedding probe delta

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
import torch
import xai_concept_leakage.data.waterbirds_loader as wb_module

from experiments.evaluate_models.eval_suite import (
    ALL_FOLDS, find_checkpoints,
    load_predictions,
    embedding_probe_delta,
    task_leakage_probe,
    run_icvl,
    run_cvl_global,
    run_intervention_curve,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRIAL_RUN        = False
RERUN_TASK       = False
RERUN_INTERV     = False
RERUN_CVL_GLOBAL = False
RERUN_EMB_DELTA  = False
RERUN_TASK_LEAK  = False
RERUN_ICVL       = False
RUN_ICVL         = True
INTERVENTION_POLICIES = ["random"]
INTERVENTION_REPEATS  = 3

WB_FOLDER  = master_folder + "results/waterbirds/"
SAVE_PATH  = master_folder + "results/results_waterbirds_suite.dict"
PLOT_DIR   = master_folder + "results/plots/waterbirds/"
LAM_C_LIST = ["lam_c0.1", "lam_c0.5", "lam_c1"]
FOLDS      = ["fold_1"] if TRIAL_RUN else ALL_FOLDS

os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
dataset_config = {
    "batch_size": 64,
    "num_workers": 0,
    "root_dir": master_folder + "data/",
    "train_augment": False,
    "weight_loss": False,
}
train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = (
    wb_module.generate_data(
        config=dataset_config, seed=42, output_dataset_vars=True
    )
)
x2c_extractor = None
print(f"Waterbirds: n_concepts={n_concepts}, n_tasks={n_tasks}")
print(f"  test groups: {np.bincount(test_dl.dataset.groups, minlength=4)}")


def worst_group_accuracy(y_pred_classes, y_true, groups):
    """Return (worst_group_acc, per_group_accs dict)."""
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
    cem_ckpts   = find_checkpoints(WB_FOLDER, "CEM_",   LAM_C, FOLDS)
    crcem_ckpts = find_checkpoints(WB_FOLDER, "CRCEM_", LAM_C, FOLDS)

    print(f"\n{'#'*60}")
    print(f"  LAM_C={LAM_C}")
    print(f"{'#'*60}")

    for label, fold_ckpts in [("cem", cem_ckpts), ("crcem", crcem_ckpts)]:
        if not fold_ckpts:
            print(f"  No {label.upper()} checkpoints for {LAM_C}, skipping.")
            continue
        results.setdefault(label, {}).setdefault(LAM_C, {})

        for fold, ckpt in fold_ckpts.items():
            r = results[label][LAM_C].get(fold, {})

            todo = {
                "task":       "task_acc"        not in r or RERUN_TASK,
                "interv":     "interv"          not in r or RERUN_INTERV,
                "cvl_global": "cvl_global"      not in r or RERUN_CVL_GLOBAL,
                "emb_delta":  "emb_probe_delta" not in r or RERUN_EMB_DELTA,
                "task_leak":  "task_leak"       not in r or RERUN_TASK_LEAK,
                "icvl":       ("icvl" not in r and RUN_ICVL) or RERUN_ICVL,
            }

            if not any(todo.values()):
                print(f"\n  Skipping {label.upper()} {fold} ({LAM_C}) — all cached")
                continue

            print(f"\n{'='*60}")
            print(f"  {label.upper()} — {fold} — {LAM_C}")
            print(f"  Computing: {[k for k, v in todo.items() if v]}")
            print(f"{'='*60}")

            print("  Loading test predictions...")
            test_preds  = load_predictions(ckpt, x2c_extractor, test_dl)
            print("  Loading train predictions...")
            train_preds = load_predictions(ckpt, x2c_extractor, train_dl)

            N_CONCEPTS = test_preds["n_concepts"]
            EMB_SIZE   = test_preds["emb_size"]

            # --- task + group accuracy ---
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

            # --- CVL global ---
            if todo["cvl_global"]:
                print(f"  Computing CVL_global...")
                r["cvl_global"] = run_cvl_global(
                    train_preds["c_mix"], test_preds["c_mix"],
                    train_preds["c_true"], test_preds["c_true"],
                    train_preds["y_true"], test_preds["y_true"],
                )
            cg = r["cvl_global"]
            print(f"  CVL_global={cg['CVL_global']:.4f}")

            # --- Embedding probe delta ---
            if todo["emb_delta"]:
                print(f"  Computing embedding probe delta...")
                r["emb_probe_delta"] = embedding_probe_delta(
                    train_preds["c_prob"], test_preds["c_prob"],
                    train_preds["c_mix"],  test_preds["c_mix"],
                    train_preds["y_true"], test_preds["y_true"],
                )
            epd = r["emb_probe_delta"]
            print(f"  Scalar={epd['acc_scalar']:.4f}  Emb={epd['acc_emb']:.4f}  Δ={epd['delta']:.4f}")

            # --- Task leakage probe ---
            if todo["task_leak"]:
                print(f"  Computing task leakage probe...")
                r["task_leak"] = task_leakage_probe(
                    train_preds["c_true"], test_preds["c_true"],
                    train_preds["c_mix"],  test_preds["c_mix"],
                    train_preds["y_true"], test_preds["y_true"],
                )
            tl = r["task_leak"]
            print(f"  y|c={tl['acc_concepts']:.4f}  y|cĈ={tl['acc_concat']:.4f}  Δ={tl['acc_delta']:.4f}")

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

            # --- ICVL ---
            if todo["icvl"]:
                print(f"  Computing ICVL ({N_CONCEPTS}×{N_CONCEPTS-1} probes)...")
                r["icvl"] = run_icvl(
                    train_preds["c_mix"], test_preds["c_mix"],
                    train_preds["c_true"], test_preds["c_true"],
                )
            if "icvl" in r:
                print(f"  ICVL={r['icvl']['ICVL']:.4f}")

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
group_names = ["LL", "LW", "WL", "WW"]
for label in ["cem", "crcem"]:
    print(f"\n  {label.upper()}")
    for lam_c in LAM_C_LIST:
        if lam_c not in results.get(label, {}):
            continue
        task_accs, worst_accs = [], []
        for fold, r in results[label][lam_c].items():
            if "task_acc" in r:
                task_accs.append(r["task_acc"])
                worst_accs.append(r["worst_group_acc"])
        if task_accs:
            print(f"    [{lam_c}]  avg_acc={np.mean(task_accs):.4f}±{np.std(task_accs):.4f}"
                  f"  worst_group={np.mean(worst_accs):.4f}±{np.std(worst_accs):.4f}")
