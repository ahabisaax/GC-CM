"""
Rerun Ridge CVL (per-concept + global, correct Ridge y→R formulation)
for CUB CEM and CRCEM. Merges into results/results_cub_suite.dict.

MLP variants and ICVL are not rerun here (MLP step 2 is unaffected by
the Ridge formulation change; ICVL is 112×111 probes — prohibitively slow).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

master_folder = os.getcwd().replace("/experiments/evaluate_models", "")
sys.path.insert(0, master_folder)
master_folder = master_folder + "/"
sys.path.insert(0, master_folder + "data/CUB200")

import numpy as np
import joblib
import data.CUB200.cub_loader as cub_data_module

from experiments.evaluate_models.eval_suite import (
    ALL_FOLDS, find_checkpoints, load_predictions,
    run_cvl, run_cvl_global,
)

RERUN      = True
CEM_FOLDER = master_folder + "results/cub_cem/"
SAVE_PATH  = master_folder + "results/results_cub_suite.dict"
LAM_C      = "lam_c0.1"
FOLDS      = ALL_FOLDS

dataset_config = {
    "dataset": "cub",
    "num_workers": 0,
    "batch_size": 256,
    "root_dir": master_folder + "data/CUB200/",
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
x2c_extractor = None
print(f"CUB: n_concepts={n_concepts}, n_tasks={n_tasks}")

cem_ckpts   = find_checkpoints(CEM_FOLDER, "CEM_",   LAM_C, FOLDS)
crcem_ckpts = find_checkpoints(CEM_FOLDER, "CRCEM_", LAM_C, FOLDS)

results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

for label, fold_ckpts in [("cem", cem_ckpts), ("crcem", crcem_ckpts)]:
    results.setdefault(label, {})

    for fold, ckpt in fold_ckpts.items():
        r = results[label].get(fold, {})
        need_cvl  = "cvl"        not in r or RERUN
        need_cvlg = "cvl_global" not in r or RERUN

        if not (need_cvl or need_cvlg):
            print(f"  Skipping {label.upper()} {fold} — all Ridge CVL cached")
            continue

        print(f"\n{'='*60}")
        print(f"  {label.upper()} — {fold}")
        print(f"{'='*60}")
        print("  Loading test predictions...")
        test_preds  = load_predictions(ckpt, x2c_extractor, test_dl)
        print("  Loading train predictions...")
        train_preds = load_predictions(ckpt, x2c_extractor, train_dl)

        N_CONCEPTS = test_preds["n_concepts"]
        EMB_SIZE   = test_preds["emb_size"]

        args = (
            train_preds["c_mix"], test_preds["c_mix"],
            train_preds["c_true"], test_preds["c_true"],
            train_preds["y_true"], test_preds["y_true"],
        )

        if need_cvl:
            print(f"  [1/2] Ridge per-concept CVL (y→R, {N_CONCEPTS} concepts × {EMB_SIZE}-dim)...")
            r["cvl"] = run_cvl(*args)
        cvl = r["cvl"]
        print(f"  CVL={cvl['CVL']:.4f}  "
              f"(min_r2={min(cvl['r2_per_concept']):.4f}  "
              f"max_r2={max(cvl['r2_per_concept']):.4f})")

        if need_cvlg:
            print(f"  [2/2] Ridge global CVL (y→R, {N_CONCEPTS*EMB_SIZE}-dim)...")
            r["cvl_global"] = run_cvl_global(*args)
        cg = r["cvl_global"]
        print(f"  CVL_global={cg['CVL_global']:.4f}  r2={cg['r2']:.4f}")

        results[label][fold] = r
        joblib.dump(results, SAVE_PATH)
        print(f"  Saved → {SAVE_PATH}")

print("\n=== Summary ===")
for label in ["cem", "crcem"]:
    if label not in results:
        continue
    print(f"\n  {label.upper()}")
    for fold, r in results[label].items():
        parts = []
        if "cvl"        in r: parts.append(f"CVL={r['cvl']['CVL']:.4f}")
        if "cvl_global" in r: parts.append(f"CVL_g={r['cvl_global']['CVL_global']:.4f}")
        print(f"    {fold}  " + "  ".join(parts))
