"""
Rerun CVL (per-concept + global, Ridge→Ridge with R²) and compute ICVL
for TabularToy CEM and CRCEM. Merges into results/results_tabulartoy_suite.dict.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

master_folder = os.getcwd().replace("/experiments/evaluate_models", "")
sys.path.insert(0, master_folder)
master_folder = master_folder + "/"


def main():
    import numpy as np
    import joblib

    from xai_concept_leakage.data.tabulartoy_auxiliary import TT_dataloaders
    from experiments.experiment_utils import get_tabulartoy_extractor_arch
    from experiments.evaluate_models.eval_suite import (
        ALL_FOLDS, find_checkpoints, load_predictions,
        run_cvl, run_cvl_global, run_icvl,
    )

    RERUN          = True   # force recompute CVL with new logistic step 2
    RESULTS_FOLDER = master_folder + "results/tabulartoy_25_10k_models_acem_shared_critic/"
    SAVE_PATH      = master_folder + "results/results_tabulartoy_suite.dict"
    LAM_C          = "lam_c0.1"
    FOLDS          = ALL_FOLDS

    save_folder = master_folder + "data/TabularToy/tabulartoy_25_10k/"
    train_dl, val_dl, test_dl = TT_dataloaders(
        save_folder, considered_concepts=["0", "1", "2"], c_logits=False, num_workers=1
    )
    x2c_extractor = get_tabulartoy_extractor_arch

    cem_ckpts   = find_checkpoints(RESULTS_FOLDER, "CEM_",   LAM_C, FOLDS)
    crcem_ckpts = find_checkpoints(RESULTS_FOLDER, "CRCEM_", LAM_C, FOLDS)

    results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

    for label, fold_ckpts in [("cem", cem_ckpts), ("crcem", crcem_ckpts)]:
        results.setdefault(label, {})

        for fold, ckpt in fold_ckpts.items():
            r = results[label].get(fold, {})
            need_cvl  = "cvl"      not in r or RERUN
            need_cvlg = "cvl_global" not in r or RERUN
            need_icvl = "icvl"     not in r or RERUN

            if not (need_cvl or need_cvlg or need_icvl):
                print(f"  Skipping {label.upper()} {fold} — all cached")
                continue

            print(f"\n{'='*60}")
            print(f"  {label.upper()} — {fold}")
            print(f"{'='*60}")
            print("  Loading predictions...")
            test_preds  = load_predictions(ckpt, x2c_extractor, test_dl)
            train_preds = load_predictions(ckpt, x2c_extractor, train_dl)
            K, m = test_preds["n_concepts"], test_preds["emb_size"]

            args = (
                train_preds["c_mix"], test_preds["c_mix"],
                train_preds["c_true"], test_preds["c_true"],
                train_preds["y_true"], test_preds["y_true"],
            )

            if need_cvl:
                print(f"  [1/3] CVL per-concept (Ridge→Ridge R², {K} concepts)...")
                r["cvl"] = run_cvl(*args)
            print(f"  CVL={r['cvl']['CVL']:.4f}  "
                  f"r2_per_concept={[f'{v:.3f}' for v in r['cvl']['r2_per_concept']]}")

            if need_cvlg:
                print(f"  [2/3] CVL global (Ridge→Ridge R², {K*m}-dim)...")
                r["cvl_global"] = run_cvl_global(*args)
            cg = r["cvl_global"]
            print(f"  CVL_global={cg['CVL_global']:.4f}  r2={cg['r2']:.4f}")

            if need_icvl:
                print(f"  [3/3] ICVL ({K}×{K-1} logistic probes)...")
                r["icvl"] = run_icvl(
                    train_preds["c_mix"], test_preds["c_mix"],
                    train_preds["c_true"], test_preds["c_true"],
                )
            print(f"  ICVL={r['icvl']['ICVL']:.4f}")

            results[label][fold] = r
            joblib.dump(results, SAVE_PATH)
            print(f"  Saved → {SAVE_PATH}")

    print("\n=== Summary ===")
    for label in ["cem", "crcem"]:
        if label not in results: continue
        print(f"\n  {label.upper()}")
        for fold, r in results[label].items():
            parts = []
            if "cvl"        in r: parts.append(f"CVL={r['cvl']['CVL']:.4f}")
            if "cvl_global" in r: parts.append(f"CVL_g={r['cvl_global']['CVL_global']:.4f}")
            if "icvl"       in r: parts.append(f"ICVL={r['icvl']['ICVL']:.4f}")
            print(f"    {fold}  " + "  ".join(parts))


if __name__ == "__main__":
    main()
