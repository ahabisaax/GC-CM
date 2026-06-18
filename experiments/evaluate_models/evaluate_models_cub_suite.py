"""
CEM/CRCEM evaluation suite for CUB-200 (112 concepts, 200 classes).

Metrics
-------
- Task accuracy
- Error-set analysis (task acc bucketed by # wrong concepts)
- Probe accuracy vs PCA components (averaged across all concepts, 1–max_comp)
- Inter-concept leakage probe (K×K acc and BCE matrices)

CTL (KSG MI) is skipped — 112 concepts × 1792-dim embeddings × 3 repeats
makes KSG infeasibly slow (hours per fold).
ICL (KSG pairwise) is skipped — 6216 pairs, prohibitively slow.
Per-concept CVL and per-concept PCA scatter are skipped — 112 concepts makes them noisy.
CVL_global (single regression over full [N, K*m]) is computed instead.

Set TRIAL_RUN = True  → fold_1 only (fast sanity check)
Set TRIAL_RUN = False → all five folds

Set RUN_ICL_PROBE = False to skip the inter-concept probe (112×111 fits per fold,
~5-10 min parallelised; baseline is cached after the first run).

Results saved to  results/results_cub_suite.dict
Plots saved to    results/plots/cub/
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
    error_set_analysis, print_error_set,
    probe_accuracy_vs_pca,
    plot_probe_curve,
    embedding_probe_delta,
    task_leakage_probe,
    run_icvl,
    run_cvl, run_cvl_global,
    run_intervention_curve,
    adversarial_concept_flip,
    interconcept_probe_analysis, print_icl_probe_summary,
    run_ois_nis_cas,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRIAL_RUN      = False   # False → run all 5 folds
# Per-metric rerun flags — set True to recompute that metric even if cached
RERUN_TASK       = False
RERUN_ERRSET     = False
RERUN_PROBE      = False
RERUN_CVL_GLOBAL = False
RERUN_CVL        = False
RERUN_EMB_DELTA  = False
RERUN_TASK_LEAK  = False
RERUN_INTERV     = False
RERUN_ADV_FLIP   = False
RERUN_ICL_PROBE  = False
RUN_ICL_PROBE    = False   # False → skip inter-concept probe for CUB
RERUN_OIS_NIS_CAS = False
RUN_ICVL   = True
RERUN_ICVL = False
INTERVENTION_POLICIES = ["random"]
INTERVENTION_REPEATS  = 3
CEM_FOLDER     = master_folder + "results/cub_cem/"
SAVE_PATH      = master_folder + "results/results_cub_suite.dict"
PLOT_DIR       = master_folder + "results/plots/cub/"
BASELINE_CACHE = master_folder + "results/cache/cub_icl_baseline.joblib"
LAM_C_LIST     = ["lam_c0.1", "lam_c0.5", "lam_c1"]
FOLDS          = ["fold_1"] if TRIAL_RUN else ALL_FOLDS
MAX_PCA_COMP   = 32

os.makedirs(PLOT_DIR, exist_ok=True)

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
train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = (
    cub_data_module.generate_data(
        config=dataset_config, seed=42,
        output_dataset_vars=True,
        root_dir=dataset_config["root_dir"],
    )
)
x2c_extractor = None
print(f"CUB: n_concepts={n_concepts}, n_tasks={n_tasks}")

# ---------------------------------------------------------------------------
# Metrics — iterate over lambda, model type, fold
# ---------------------------------------------------------------------------
results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

# Migrate flat λ=0.1 results (old format) into nested structure
for label in ["cem", "crcem"]:
    if label in results and "fold_1" in results[label]:
        print(f"  Migrating existing {label} results → lam_c0.1")
        results[label] = {"lam_c0.1": results[label]}

for LAM_C in LAM_C_LIST:
    cem_ckpts   = find_checkpoints(CEM_FOLDER, "CEM_",   LAM_C, FOLDS)
    crcem_ckpts = find_checkpoints(CEM_FOLDER, "CRCEM_", LAM_C, FOLDS)

    print(f"\n{'#'*60}")
    print(f"  LAM_C={LAM_C}  TRIAL_RUN={TRIAL_RUN}  folds={FOLDS}")
    print(f"{'#'*60}")
    for fold, path in cem_ckpts.items():
        print(f"  CEM   {fold}: {os.path.relpath(path, master_folder)}")
    for fold, path in crcem_ckpts.items():
        print(f"  CRCEM {fold}: {os.path.relpath(path, master_folder)}")

    for label, fold_ckpts in [("cem", cem_ckpts), ("crcem", crcem_ckpts)]:
        results.setdefault(label, {}).setdefault(LAM_C, {})

        for fold, ckpt in fold_ckpts.items():
            r = results[label][LAM_C].get(fold, {})

            todo = {
                "task":       "task_acc"        not in r or RERUN_TASK,
                "errset":     "error_set"       not in r or RERUN_ERRSET,
                "probe":      "probe_acc"       not in r or RERUN_PROBE,
                "cvl_global": "cvl_global"      not in r or RERUN_CVL_GLOBAL,
                "cvl":        "cvl"             not in r or RERUN_CVL,
                "emb_delta":  "emb_probe_delta" not in r or RERUN_EMB_DELTA,
                "task_leak":  "task_leak"       not in r or RERUN_TASK_LEAK,
                "interv":     "interv"          not in r or RERUN_INTERV,
                "adv_flip":   "adv_flip"        not in r or RERUN_ADV_FLIP,
                "icl_probe":   ("icl_probe" not in r and RUN_ICL_PROBE) or RERUN_ICL_PROBE,
                "ois_nis_cas": "ois_nis_cas" not in r or RERUN_OIS_NIS_CAS,
                "icvl":        ("icvl" not in r and RUN_ICVL) or RERUN_ICVL,
            }

            if not any(todo.values()):
                print(f"\n  Skipping {label.upper()} {fold} — all metrics cached")
                continue

            print(f"\n{'='*60}")
            print(f"  {label.upper()} — {fold}")
            print(f"  Computing: {[k for k, v in todo.items() if v]}")
            print(f"{'='*60}")

            print("  Loading test predictions...")
            test_preds  = load_predictions(ckpt, x2c_extractor, test_dl)
            print("  Loading train predictions...")
            train_preds = load_predictions(ckpt, x2c_extractor, train_dl)

            N_CONCEPTS = test_preds["n_concepts"]
            EMB_SIZE   = test_preds["emb_size"]
            max_comp   = min(MAX_PCA_COMP, EMB_SIZE)

            # --- task accuracy ---
            if todo["task"]:
                r["task_acc"] = float((test_preds["y_pred"].argmax(-1) == test_preds["y_true"]).mean())
                r["c_acc"]    = float((test_preds["c_pred"] == test_preds["c_true"]).all(axis=1).mean())
            print(f"  task_acc={r['task_acc']:.4f}  c_acc={r['c_acc']:.4f}")

            # --- error-set analysis ---
            if todo["errset"]:
                r["error_set"] = error_set_analysis(
                    test_preds["c_pred"], test_preds["c_true"],
                    test_preds["y_pred"], test_preds["y_true"],
                )
            print_error_set(r["error_set"], label=f"{label} {fold}")

            # CTL (KSG MI) skipped — infeasible for 112 concepts × 1792-dim

            # --- Probe accuracy vs PCA ---
            if todo["probe"]:
                print(f"  Computing probe accuracy curve ({max_comp} components, {N_CONCEPTS} concepts)...")
                probe_acc = probe_accuracy_vs_pca(
                    train_preds["c_mix"], test_preds["c_mix"],
                    train_preds["y_true"], test_preds["y_true"],
                    max_components=max_comp,
                )
                r["probe_acc"] = probe_acc.tolist()
            probe_acc = np.array(r["probe_acc"])
            print(f"  Probe acc at {min(MAX_PCA_COMP, EMB_SIZE)} components (avg over concepts): "
                  f"{probe_acc[:, -1].mean():.4f}")

            # --- CVL global (single regression over full [N, K*m] embedding) ---
            if todo["cvl_global"]:
                print(f"  Computing CVL_global ({N_CONCEPTS} concepts × {EMB_SIZE}-dim)...")
                r["cvl_global"] = run_cvl_global(
                    train_preds["c_mix"], test_preds["c_mix"],
                    train_preds["c_true"], test_preds["c_true"],
                    train_preds["y_true"], test_preds["y_true"],
                )
            cg = r["cvl_global"]
            print(f"  CVL_global={cg['CVL_global']:.4f}  r2={cg.get('r2', cg.get('acc', 'n/a')):.4f}")

            # --- Per-concept CVL (mean R² across all K concepts) ---
            if todo["cvl"]:
                print(f"  Computing per-concept CVL ({N_CONCEPTS} concepts × {EMB_SIZE}-dim)...")
                r["cvl"] = run_cvl(
                    train_preds["c_mix"], test_preds["c_mix"],
                    train_preds["c_true"], test_preds["c_true"],
                    train_preds["y_true"], test_preds["y_true"],
                )
            cv = r["cvl"]
            print(f"  CVL={cv['CVL']:.4f}  (mean over {N_CONCEPTS} concepts)")

            # --- Embedding probe delta ---
            if todo["emb_delta"]:
                print(f"  Computing embedding probe delta (scalar vs embedding, {N_CONCEPTS} concepts)...")
                epd = embedding_probe_delta(
                    train_preds["c_prob"], test_preds["c_prob"],
                    train_preds["c_mix"],  test_preds["c_mix"],
                    train_preds["y_true"], test_preds["y_true"],
                )
                r["emb_probe_delta"] = epd
            epd = r["emb_probe_delta"]
            print(f"  Scalar acc: {epd['acc_scalar']:.4f}")
            print(f"  Emb    acc: {epd['acc_emb']:.4f}")
            print(f"  Delta:      {epd['delta']:.4f}")

            # --- Task leakage probe: y|c vs y|[c, Ĉ] ---
            if todo["task_leak"]:
                print(f"  Computing task leakage probe (y|c vs y|[c,Ĉ])...")
                r["task_leak"] = task_leakage_probe(
                    train_preds["c_true"], test_preds["c_true"],
                    train_preds["c_mix"],  test_preds["c_mix"],
                    train_preds["y_true"], test_preds["y_true"],
                )
            tl = r["task_leak"]
            print(f"  y|c  acc={tl['acc_concepts']:.4f}  bce={tl['bce_concepts']:.4f}")
            print(f"  y|cĈ acc={tl['acc_concat']:.4f}  bce={tl['bce_concat']:.4f}")
            print(f"  Δacc={tl['acc_delta']:.4f}  ΔBCE={tl['bce_delta']:.4f}")

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
                print(f"  Intervention [{policy}]: 0 concepts={mean_curve[0]:.4f} → all={mean_curve[-1]:.4f}")

            # --- Adversarial concept flip ---
            if todo["adv_flip"]:
                print(f"  Computing adversarial concept flip...")
                r["adv_flip"] = adversarial_concept_flip(
                    ckpt, x2c_extractor, train_dl, test_preds
                )
            af = r["adv_flip"]
            print(f"  Adv flip: k=0→{af['acc_0']:.4f}  k=all→{af['acc_all']:.4f}"
                  f"  drop={af['acc_0']-af['acc_all']:.4f}")

            # --- Inter-concept leakage probe ---
            if todo["icl_probe"]:
                print(f"  Computing inter-concept leakage probes ({N_CONCEPTS}×{N_CONCEPTS})...")
                r["icl_probe"] = interconcept_probe_analysis(
                    train_preds["c_mix"], test_preds["c_mix"],
                    train_preds["c_true"], test_preds["c_true"],
                    baseline_cache_path=BASELINE_CACHE,
                )
            if "icl_probe" in r:
                print_icl_probe_summary(r["icl_probe"], label=f"{label} {fold}")

            # --- OIS / NIS / CAS ---
            if todo["ois_nis_cas"]:
                print(f"  Computing OIS / NIS / CAS (CUB: 112 concepts, may be slow)...")
                r["ois_nis_cas"] = run_ois_nis_cas(ckpt, x2c_extractor, train_dl, test_dl)
            onc = r["ois_nis_cas"]
            print(f"  OIS={onc['ois']:.4f}  NIS={onc['nis']:.4f}  CAS={onc['cas']:.4f}")

            # --- ICVL (skipped by default for CUB — 112×111 probes) ---
            if todo["icvl"]:
                print(f"  Computing ICVL ({N_CONCEPTS}×{N_CONCEPTS-1} logistic probes)...")
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
# Probe curve plot (averaged across all concepts) — use lam_c0.1 / fold_1
# ---------------------------------------------------------------------------
print("\nSaving probe accuracy plot...")
_vis_lam = "lam_c0.1"
_vis_cem_ckpts   = find_checkpoints(CEM_FOLDER, "CEM_",   _vis_lam, FOLDS)
_vis_crcem_ckpts = find_checkpoints(CEM_FOLDER, "CRCEM_", _vis_lam, FOLDS)
if (_vis_lam in results.get("cem", {}) and _vis_lam in results.get("crcem", {})
        and _vis_cem_ckpts and _vis_crcem_ckpts):
    first_fold  = next(iter(_vis_cem_ckpts))
    cem_probe   = np.array(results["cem"][_vis_lam][first_fold]["probe_acc"])
    crcem_probe = np.array(results["crcem"][_vis_lam][first_fold]["probe_acc"])
    plot_probe_curve(
        cem_probe, crcem_probe,
        title=f"CUB — probe accuracy averaged over {n_concepts} concepts ({first_fold})",
        save_path=PLOT_DIR + "probe_curve_avg.png",
        average=True,
    )

# ---------------------------------------------------------------------------
# Save and summary
# ---------------------------------------------------------------------------
joblib.dump(results, SAVE_PATH)
print(f"\nResults saved → {SAVE_PATH}")

print("\n=== Summary ===")
for label in ["cem", "crcem"]:
    print(f"\n  {label.upper()}")
    for lam_c in LAM_C_LIST:
        if lam_c not in results.get(label, {}):
            continue
        print(f"    [{lam_c}]")
        for fold, r in results[label][lam_c].items():
            probe = np.array(r["probe_acc"])
            n_wrong  = sum(v["n_samples"] for k, v in r["error_set"].items() if k > 0)
            acc_wrong = np.mean([v["task_acc"] for k, v in r["error_set"].items() if k > 0])
            line = (f"      {fold}  task={r['task_acc']:.4f}"
                    f"  probe@{min(MAX_PCA_COMP, r['probe_acc'][0].__len__())}={probe[:,-1].mean():.4f}"
                    f"  err_acc={acc_wrong:.4f}/{n_wrong}")
            if "cvl_global" in r:
                line += f"  CVL_g={r['cvl_global']['CVL_global']:.4f}"
            if "cvl" in r:
                line += f"  CVL={r['cvl']['CVL']:.4f}"
            if "icl_probe" in r:
                K = n_concepts
                mask = ~np.eye(K, dtype=bool)
                line += f"  ICL_Δacc={r['icl_probe']['acc_diff'][mask].mean():.4f}"
            if "emb_probe_delta" in r:
                line += f"  EmbΔ={r['emb_probe_delta']['delta']:.4f}"
            if "icvl" in r:
                line += f"  ICVL={r['icvl']['ICVL']:.4f}"
            if "ois_nis_cas" in r:
                onc = r["ois_nis_cas"]
                line += f"  OIS={onc['ois']:.4f}  NIS={onc['nis']:.4f}  CAS={onc['cas']:.4f}"
            print(line)
