"""
Standalone: compute all CVL variants for CUB (Ridge and MLP, per-concept and global).
Results are merged into results/results_cub_suite.dict.

Run from the project root:
  python experiments/evaluate_models/compute_cub_cvl.py
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
    ALL_FOLDS, find_checkpoints,
    load_predictions,
    run_cvl, run_cvl_global,
    run_cvl_mlp, run_cvl_global_mlp,
    run_cvl_mlp2, run_cvl_global_mlp2,
    run_icvl,
)

RERUN      = True    # recompute all CVL variants
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

VARIANTS = [
    ("cvl",            "cvl",            "Ridge per-concept"),
    ("cvl_global",     "cvl_global",     "Ridge global"),
    ("cvl_mlp",        "cvl_mlp",        "MLP(Ridge) per-concept"),
    ("cvl_global_mlp", "cvl_global_mlp", "MLP(Ridge) global"),
    ("cvl_mlp2",       "cvl_mlp2",       "MLP(MLP) per-concept"),
    ("cvl_global_mlp2","cvl_global_mlp2","MLP(MLP) global"),
]
RERUN_ICVL = True

for label, fold_ckpts in [("cem", cem_ckpts), ("crcem", crcem_ckpts)]:
    results.setdefault(label, {})

    for fold, ckpt in fold_ckpts.items():
        r = results[label].get(fold, {})
        todo = {key: (key not in r or RERUN) for key, _, _ in VARIANTS}
        need_icvl = "icvl" not in r or RERUN_ICVL

        if not any(todo.values()) and not need_icvl:
            print(f"  Skipping {label.upper()} {fold} — all CVL/ICVL variants cached")
            continue

        print(f"\n{'='*60}")
        print(f"  {label.upper()} — {fold}")
        print(f"  Computing: {[desc for key, _, desc in VARIANTS if todo[key]]}")
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

        if todo["cvl"]:
            print(f"  [1/6] Ridge per-concept CVL ({N_CONCEPTS} concepts × {EMB_SIZE}-dim)...")
            r["cvl"] = run_cvl(*args)
        print(f"  CVL={r['cvl']['CVL']:.4f}  "
              f"(min_r2={min(r['cvl']['r2_per_concept']):.4f}  "
              f"max_r2={max(r['cvl']['r2_per_concept']):.4f})")

        if todo["cvl_global"]:
            print(f"  [2/6] Ridge global CVL ({N_CONCEPTS*EMB_SIZE}-dim)...")
            r["cvl_global"] = run_cvl_global(*args)
        cg = r["cvl_global"]
        print(f"  CVL_global={cg['CVL_global']:.4f}  r2={cg['r2']:.4f}")

        if todo["cvl_mlp"]:
            print(f"  [3/6] MLP(Ridge) per-concept CVL...")
            r["cvl_mlp"] = run_cvl_mlp(*args)
        cm = r["cvl_mlp"]
        print(f"  CVL_mlp={cm['CVL_mlp']:.4f}  baseline={cm['baseline_acc']:.4f}  "
              f"mean_acc={float(np.mean(cm['acc_per_concept'])):.4f}")

        if todo["cvl_global_mlp"]:
            print(f"  [4/6] MLP(Ridge) global CVL (PCA→128)...")
            r["cvl_global_mlp"] = run_cvl_global_mlp(*args, pca_components=128)
        cgm = r["cvl_global_mlp"]
        print(f"  CVL_global_mlp={cgm['CVL_global_mlp']:.4f}  acc={cgm['acc']:.4f}  "
              f"baseline={cgm['baseline_acc']:.4f}")

        if todo["cvl_mlp2"]:
            print(f"  [5/6] MLP(MLP) per-concept CVL...")
            r["cvl_mlp2"] = run_cvl_mlp2(*args)
        cm2 = r["cvl_mlp2"]
        print(f"  CVL_mlp2={cm2['CVL_mlp2']:.4f}  baseline={cm2['baseline_acc']:.4f}  "
              f"mean_acc={float(np.mean(cm2['acc_per_concept'])):.4f}")

        if todo["cvl_global_mlp2"]:
            print(f"  [6/6] MLP(MLP) global CVL (PCA→128)...")
            r["cvl_global_mlp2"] = run_cvl_global_mlp2(*args, pca_components=128)
        cgm2 = r["cvl_global_mlp2"]
        print(f"  CVL_global_mlp2={cgm2['CVL_global_mlp2']:.4f}  acc={cgm2['acc']:.4f}  "
              f"baseline={cgm2['baseline_acc']:.4f}")

        if need_icvl:
            print(f"  [7/?] ICVL ({N_CONCEPTS}×{N_CONCEPTS-1} Ridge probes)...")
            r["icvl"] = run_icvl(
                train_preds["c_mix"], test_preds["c_mix"],
                train_preds["c_true"], test_preds["c_true"],
            )
        print(f"  ICVL={r['icvl']['ICVL']:.4f}")

        results[label][fold] = r
        joblib.dump(results, SAVE_PATH)
        print(f"  Saved → {SAVE_PATH}")

        # Print fold summary immediately so we can see results as they arrive
        print(f"\n  *** {label.upper()} {fold} summary ***")
        print(f"    Ridge:     CVL={r['cvl']['CVL']:.4f}  CVL_g={r['cvl_global']['CVL_global']:.4f}")
        print(f"    MLP(R):    CVL={r['cvl_mlp']['CVL_mlp']:.4f}  CVL_g={r['cvl_global_mlp']['CVL_global_mlp']:.4f}")
        print(f"    MLP(MLP):  CVL={r['cvl_mlp2']['CVL_mlp2']:.4f}  CVL_g={r['cvl_global_mlp2']['CVL_global_mlp2']:.4f}")
        print(f"    ICVL:      {r['icvl']['ICVL']:.4f}")

print("\n=== Final Summary ===")
for label in ["cem", "crcem"]:
    if label not in results:
        continue
    print(f"\n  {label.upper()}")
    for fold, r in results[label].items():
        parts = []
        if "cvl"            in r: parts.append(f"R={r['cvl']['CVL']:.4f}")
        if "cvl_global"     in r: parts.append(f"R_g={r['cvl_global']['CVL_global']:.4f}")
        if "cvl_mlp"        in r: parts.append(f"M1={r['cvl_mlp']['CVL_mlp']:.4f}")
        if "cvl_global_mlp" in r: parts.append(f"M1_g={r['cvl_global_mlp']['CVL_global_mlp']:.4f}")
        if "cvl_mlp2"       in r: parts.append(f"M2={r['cvl_mlp2']['CVL_mlp2']:.4f}")
        if "cvl_global_mlp2"in r: parts.append(f"M2_g={r['cvl_global_mlp2']['CVL_global_mlp2']:.4f}")
        if "icvl"           in r: parts.append(f"ICVL={r['icvl']['ICVL']:.4f}")
        print(f"    {fold}  " + "  ".join(parts))
