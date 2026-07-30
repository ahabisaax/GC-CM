"""
Compute global CVL variants for dSprites (Ridge, MLP(Ridge), MLP(MLP)).
Merges results into results/results_dsprites_suite.dict.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

master_folder = os.getcwd().replace("/experiments/evaluate_models", "")
sys.path.insert(0, master_folder)
master_folder = master_folder + "/"


def main():
    import numpy as np
    import joblib

    from xai_concept_leakage.data.dsprites_auxiliary import dsprites_dataloaders
    from experiments.experiment_utils import get_dsprites_extractor_arch
    from experiments.evaluate_models.eval_suite import (
        ALL_FOLDS, find_checkpoints,
        load_predictions,
        run_cvl_global, run_cvl_global_mlp, run_cvl_global_mlp2,
    )

    RERUN          = False
    CEM_FOLDER     = master_folder + "results/dsprites_cem/"
    CRCEM_FOLDER   = master_folder + "results/dsprites_acem_shared_critic/"
    SAVE_PATH      = master_folder + "results/results_dsprites_suite.dict"
    LAM_C          = "lam_c0.1"
    FOLDS          = ALL_FOLDS

    corr = 0
    path_dataset = master_folder + f"data/dsprites/dsprites_dep_{corr}.npz"
    train_dl, val_dl, test_dl = dsprites_dataloaders(
        path_dataset, val_ratio=0.1, num_workers=1, batch_size=256
    )
    x2c_extractor = get_dsprites_extractor_arch

    cem_ckpts   = find_checkpoints(CEM_FOLDER,   "CEM_",   LAM_C, FOLDS)
    crcem_ckpts = find_checkpoints(CRCEM_FOLDER, "CRCEM_", LAM_C, FOLDS)

    results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

    for label, fold_ckpts in [("cem", cem_ckpts), ("crcem", crcem_ckpts)]:
        results.setdefault(label, {})

        for fold, ckpt in fold_ckpts.items():
            r = results[label].get(fold, {})
            need_g     = "cvl_global"      not in r or RERUN
            need_g_mlp = "cvl_global_mlp"  not in r or RERUN
            need_g_m2  = "cvl_global_mlp2" not in r or RERUN

            if not (need_g or need_g_mlp or need_g_m2):
                print(f"  Skipping {label.upper()} {fold} — all global CVL cached")
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

            if need_g:
                print(f"  [1/3] Ridge global CVL ({K*m}-dim)...")
                r["cvl_global"] = run_cvl_global(*args)
            cg = r["cvl_global"]
            print(f"  CVL_global={cg['CVL_global']:.4f}  r2={cg['r2']:.4f}")

            if need_g_mlp:
                print(f"  [2/3] MLP(Ridge) global CVL (no PCA, {K*m}-dim)...")
                r["cvl_global_mlp"] = run_cvl_global_mlp(*args, pca_components=None)
            cgm = r["cvl_global_mlp"]
            print(f"  CVL_global_mlp={cgm['CVL_global_mlp']:.4f}  acc={cgm['acc']:.4f}  "
                  f"baseline={cgm['baseline_acc']:.4f}")

            if need_g_m2:
                print(f"  [3/3] MLP(MLP) global CVL (no PCA, {K*m}-dim)...")
                r["cvl_global_mlp2"] = run_cvl_global_mlp2(*args, pca_components=None)
            cgm2 = r["cvl_global_mlp2"]
            print(f"  CVL_global_mlp2={cgm2['CVL_global_mlp2']:.4f}  acc={cgm2['acc']:.4f}  "
                  f"baseline={cgm2['baseline_acc']:.4f}")

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
            if "cvl_global"      in r: parts.append(f"Ridge={r['cvl_global']['CVL_global']:.4f}")
            if "cvl_global_mlp"  in r: parts.append(f"MLP(R)={r['cvl_global_mlp']['CVL_global_mlp']:.4f}")
            if "cvl_global_mlp2" in r: parts.append(f"MLP(M)={r['cvl_global_mlp2']['CVL_global_mlp2']:.4f}")
            if "cvl"             in r: parts.append(f"CVL_perconcept={r['cvl']['CVL']:.4f}")
            print(f"    {fold}  " + "  ".join(parts))


if __name__ == "__main__":
    main()
