"""
Post-hoc KSG CTL and ICL for CUB-200 CEM / GC-CEM.

Usage (from project root):
    # Trial: fold_1 only, lam_c0.1, both models
    python experiments/evaluate_models/compute_cub_ctl_icl.py

    # All folds, specific model and lambda
    python experiments/evaluate_models/compute_cub_ctl_icl.py --all_folds --model crcem --lam_c lam_c0.5

    # Skip ICL (much slower — 6216 pairs per fold)
    python experiments/evaluate_models/compute_cub_ctl_icl.py --no_icl

Results cached to: results/results_cub_ctl_icl.dict
Structure: {model_key: {lam_c: {fold: {"ctl_mean": float, "ctl_se": float,
                                       "icl_mean": float, "icl_se": float}}}}
"""
import os, sys, argparse, warnings, time
warnings.filterwarnings("ignore")

master_folder = os.getcwd().replace("/experiments/evaluate_models", "")
sys.path.insert(0, master_folder)
sys.path.insert(0, master_folder + "/data/CUB200")
master_folder = master_folder + "/"

import numpy as np
import joblib
import data.CUB200.cub_loader as cub_data_module

from experiments.evaluate_models.eval_suite import (
    ALL_FOLDS, find_checkpoints, load_predictions,
)
from xai_concept_leakage.metrics.mutual_information import compute_mi_matrix_parallel

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--all_folds", action="store_true",
                    help="Run all 5 folds (default: fold_1 only)")
parser.add_argument("--model", choices=["cem", "crcem", "both"], default="both")
parser.add_argument("--lam_c", choices=["lam_c0.1", "lam_c0.5", "lam_c1", "all"],
                    default="all")
parser.add_argument("--no_icl", action="store_true",
                    help="Skip ICL (6216 pairs — may take hours)")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--n_neighbors", type=int, default=3)
parser.add_argument("--rerun", action="store_true",
                    help="Recompute even if already cached")
args = parser.parse_args()

FOLDS   = ALL_FOLDS if args.all_folds else ["fold_1"]
MODELS  = ["cem", "crcem"] if args.model == "both" else [args.model]
LAM_CS  = ["lam_c0.1", "lam_c0.5", "lam_c1"] if args.lam_c == "all" else [args.lam_c]

CEM_FOLDER = master_folder + "results/cub_cem/"
SAVE_PATH  = master_folder + "results/results_cub_ctl_icl.dict"

print(f"Models: {MODELS}  |  Lambdas: {LAM_CS}  |  Folds: {FOLDS}")
print(f"ICL: {'skipped' if args.no_icl else 'enabled'}  |  repeats={args.repeats}")
print(f"Cache: {SAVE_PATH}\n")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
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
print("Loading CUB data...")
train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = (
    cub_data_module.generate_data(
        config=dataset_config, seed=42,
        output_dataset_vars=True,
        root_dir=dataset_config["root_dir"],
    )
)
print(f"n_concepts={n_concepts}, n_tasks={n_tasks}\n")

# ---------------------------------------------------------------------------
# Load cache
# ---------------------------------------------------------------------------
results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
for lam_c in LAM_CS:
    ckpts = {
        "cem":   find_checkpoints(CEM_FOLDER, "CEM_",   lam_c, FOLDS),
        "crcem": find_checkpoints(CEM_FOLDER, "CRCEM_", lam_c, FOLDS),
    }

    for model_key in MODELS:
        results.setdefault(model_key, {}).setdefault(lam_c, {})
        fold_ckpts = ckpts[model_key]

        for fold, ckpt in fold_ckpts.items():
            cached = results[model_key][lam_c].get(fold, {})
            need_ctl = "ctl_mean" not in cached or args.rerun
            need_icl = (not args.no_icl) and ("icl_mean" not in cached or args.rerun)

            if not need_ctl and not need_icl:
                r = cached
                print(f"  {model_key} {lam_c} {fold}: cached  "
                      f"CTL={r['ctl_mean']:.4f}±{r['ctl_se']:.4f}  "
                      f"ICL={r.get('icl_mean', float('nan')):.4f}")
                continue

            print(f"\n{'='*60}")
            print(f"  {model_key.upper()} | {lam_c} | {fold}")
            print(f"  Checkpoint: {os.path.relpath(ckpt, master_folder)}")

            t0 = time.time()
            print("  Loading test predictions...")
            preds = load_predictions(ckpt, None, test_dl)
            c_mix_flat = preds["c_mix"].reshape(len(preds["y_true"]), -1)
            print(f"  c_mix shape: {preds['c_mix'].shape}  "
                  f"(N={len(preds['y_true'])}, K={n_concepts}, m={preds['emb_size']})")

            r = dict(cached)

            n_jobs = os.cpu_count() or 1
            # c_true is binary [N, K] — reshape to [N, K, 1] for compute_mi_matrix_parallel
            c_true_flat = preds["c_true"].reshape(len(preds["y_true"]), n_concepts, 1)
            c_true_flat = c_true_flat.reshape(len(preds["y_true"]), -1)

            if need_ctl:
                print(f"  Computing CTL ({n_concepts} concepts, {args.repeats} repeats, "
                      f"{n_jobs} jobs)...")
                t1 = time.time()
                ctl_runs, ctl_true_runs = [], []
                for _ in range(args.repeats):
                    # MI(c_learned, y) — [K] vector (one per concept)
                    mi_learned_y = compute_mi_matrix_parallel(
                        c_mix_flat, d=preds["y_true"],
                        n_neighbors=args.n_neighbors,
                        normalise=True, flatten=False, n_jobs=n_jobs,
                    )
                    # MI(c_true, y) — [K] vector
                    mi_true_y = compute_mi_matrix_parallel(
                        c_true_flat, d=preds["y_true"],
                        n_neighbors=args.n_neighbors,
                        normalise=True, flatten=False, n_jobs=n_jobs,
                    )
                    ctl_runs.append(np.mean(np.maximum(0, mi_learned_y - mi_true_y)))
                    ctl_true_runs.append(float(np.mean(mi_true_y)))
                ctl_mean = float(np.mean(ctl_runs))
                ctl_se   = float(np.std(ctl_runs) / max(1, np.sqrt(len(ctl_runs) - 1)))
                r["ctl_mean"]      = ctl_mean
                r["ctl_se"]        = ctl_se
                r["ctl_true_mean"] = float(np.mean(ctl_true_runs))
                print(f"  CTL = {ctl_mean:.4f} ± {ctl_se:.4f}  [{time.time()-t1:.0f}s]")

            if need_icl:
                n_pairs = n_concepts * (n_concepts - 1) // 2
                print(f"  Computing ICL ({n_pairs} pairs, {args.repeats} repeats, "
                      f"{n_jobs} jobs)...")
                t1 = time.time()
                icl_runs = []
                for _ in range(args.repeats):
                    # MI(c_learned pairs) — lower-tri flattened
                    mi_learned = compute_mi_matrix_parallel(
                        c_mix_flat,
                        n_neighbors=args.n_neighbors,
                        normalise=True, flatten=True, n_jobs=n_jobs,
                    )
                    # MI(c_true pairs)
                    mi_true = compute_mi_matrix_parallel(
                        c_true_flat,
                        n_neighbors=args.n_neighbors,
                        normalise=True, flatten=True, n_jobs=n_jobs,
                    )
                    icl_runs.append(float(np.mean(np.maximum(0, mi_learned - mi_true))))
                icl_mean = float(np.mean(icl_runs))
                icl_se   = float(np.std(icl_runs) / max(1, np.sqrt(len(icl_runs) - 1)))
                r["icl_mean"] = icl_mean
                r["icl_se"]   = icl_se
                print(f"  ICL = {icl_mean:.4f} ± {icl_se:.4f}  [{time.time()-t1:.0f}s]")

            r["_elapsed_s"] = time.time() - t0
            results[model_key][lam_c][fold] = r
            joblib.dump(results, SAVE_PATH)
            print(f"  Saved → {SAVE_PATH}  (total {r['_elapsed_s']:.0f}s)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
for model_key in ["cem", "crcem"]:
    for lam_c in LAM_CS:
        for fold in FOLDS:
            r = results.get(model_key, {}).get(lam_c, {}).get(fold)
            if r:
                icl_str = f"  ICL={r['icl_mean']:.4f}±{r['icl_se']:.4f}" if "icl_mean" in r else ""
                print(f"  {model_key:<8} {lam_c:<10} {fold}  "
                      f"CTL={r['ctl_mean']:.4f}±{r['ctl_se']:.4f}{icl_str}")
