"""
CEM/CRCEM evaluation suite for TabularToy (cov=0.25, 10k samples, 3 concepts).

Metrics
-------
- Task accuracy
- Error-set analysis (task acc bucketed by # wrong concepts)
- CTL  (concepts-task MI, per concept)
- ICL  (inter-concept MI, scalar)
- Probe accuracy vs PCA components (per concept, CEM vs CRCEM)
- PCA scatter of concept embeddings coloured by y (one concept)
- CVL  (concept variance leakage, per concept)
- Inter-concept leakage probe (K×K acc and BCE matrices)

Set TRIAL_RUN = True  → fold_1 only (fast sanity check)
Set TRIAL_RUN = False → all five folds

Results saved to  results/results_tabulartoy_suite.dict
Plots saved to    results/plots/tabulartoy/
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

master_folder = os.getcwd().replace("/experiments/evaluate_models", "")
sys.path.insert(0, master_folder)
master_folder = master_folder + "/"

import numpy as np
import joblib

from xai_concept_leakage.data.tabulartoy_auxiliary import TT_dataloaders
from experiments.experiment_utils import get_tabulartoy_extractor_arch
from experiments.evaluate_models.eval_suite import (
    ALL_FOLDS, find_checkpoints,
    load_predictions,
    error_set_analysis, print_error_set,
    compute_ctl, compute_icl,
    probe_accuracy_vs_pca,
    pca_scatter_concept,
    plot_probe_curve,
    run_icvl,
    run_cvl, run_cvl_global,
    embedding_probe_delta,
    task_leakage_probe,
    run_intervention_curve,
    adversarial_concept_flip,
    interconcept_probe_analysis, print_icl_probe_summary,
    prototype_alignment_plot,
    run_ois_nis_cas,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRIAL_RUN         = False  # False → run all 5 folds
# Per-metric rerun flags — set True to recompute that metric even if cached
RERUN_TASK        = False
RERUN_ERRSET      = False
RERUN_CTL         = False
RERUN_ICL_KSG     = False
RERUN_PROBE       = False
RERUN_CVL         = False
RERUN_CVL_GLOBAL  = False
RERUN_EMB_DELTA   = False
RERUN_TASK_LEAK   = False
RERUN_INTERV      = False
RERUN_ADV_FLIP    = False
RUN_ICL_PROBE     = False
RERUN_ICL_PROBE   = False
RERUN_OIS_NIS_CAS = False
RERUN_ICVL        = False
INTERVENTION_POLICIES = ["random"]
INTERVENTION_REPEATS  = 3
RESULTS_FOLDER    = master_folder + "results/tabulartoy_25_10k_models_acem_shared_critic/"
SAVE_PATH         = master_folder + "results/results_tabulartoy_acem_suite.dict"
PLOT_DIR          = master_folder + "results/plots/tabulartoy_acem/"
BASELINE_CACHE    = master_folder + "results/cache/tabulartoy_acem_icl_baseline.joblib"
LAM_C_LIST        = ["lam_c0.1", "lam_c0.5", "lam_c1"]
FOLDS             = ["fold_1"] if TRIAL_RUN else ALL_FOLDS
MI_REPEATS        = 3
N_NEIGHBORS       = 3
PCA_SCATTER_CONCEPT = 0

os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data_folder = master_folder + "data/TabularToy/"
cov, n_samples = 0.25, 10000
considered_concepts = ["0", "1", "2"]
save_folder = data_folder + f"tabulartoy_{int(cov*100)}_{int(n_samples//1000)}k/"

train_dl, val_dl, test_dl = TT_dataloaders(
    save_folder, considered_concepts=considered_concepts, c_logits=False, num_workers=0
)
x2c_extractor = get_tabulartoy_extractor_arch

# ---------------------------------------------------------------------------
# Metrics — iterate over lambda, then model type, then fold
# ---------------------------------------------------------------------------
results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

for LAM_C in LAM_C_LIST:
  # Checkpoint discovery for this lambda
  cem_ckpts  = find_checkpoints(RESULTS_FOLDER, "CEM_",  LAM_C, FOLDS)
  acem_ckpts = find_checkpoints(RESULTS_FOLDER, "ACEM_", LAM_C, FOLDS)

  print(f"\n{'#'*60}")
  print(f"  LAM_C={LAM_C}  TRIAL_RUN={TRIAL_RUN}  folds={FOLDS}")
  print(f"{'#'*60}")
  for fold, path in cem_ckpts.items():
      print(f"  CEM  {fold}: {os.path.relpath(path, master_folder)}")
  for fold, path in acem_ckpts.items():
      print(f"  ACEM {fold}: {os.path.relpath(path, master_folder)}")

  for label, fold_ckpts in [("cem", cem_ckpts), ("acem", acem_ckpts)]:
    results.setdefault(label, {}).setdefault(LAM_C, {})

    for fold, ckpt in fold_ckpts.items():
        r = results[label][LAM_C].get(fold, {})

        todo = {
            "task":      "task_acc"        not in r or RERUN_TASK,
            "errset":    "error_set"       not in r or RERUN_ERRSET,
            "ctl":       "ctl_mean"        not in r or RERUN_CTL,
            "icl_ksg":   "icl_mean"        not in r or RERUN_ICL_KSG,
            "probe":     "probe_acc"       not in r or RERUN_PROBE,
            "cvl":        "cvl"             not in r or RERUN_CVL,
            "cvl_global": "cvl_global"      not in r or RERUN_CVL_GLOBAL,
            "emb_delta":  "emb_probe_delta" not in r or RERUN_EMB_DELTA,
            "task_leak": "task_leak"       not in r or RERUN_TASK_LEAK,
            "interv":    "interv"          not in r or RERUN_INTERV,
            "adv_flip":  "adv_flip"        not in r or RERUN_ADV_FLIP,
            "icl_probe":    ("icl_probe" not in r and RUN_ICL_PROBE) or RERUN_ICL_PROBE,
            "ois_nis_cas":  "ois_nis_cas" not in r or RERUN_OIS_NIS_CAS,
            "icvl":         "icvl"        not in r or RERUN_ICVL,
        }

        if not any(todo.values()):
            print(f"\n  Skipping {label.upper()} {fold} — all metrics cached")
            continue

        print(f"\n{'='*60}")
        print(f"  {label.upper()} — {fold}")
        print(f"  Computing: {[k for k, v in todo.items() if v]}")
        print(f"{'='*60}")

        test_preds  = load_predictions(ckpt, x2c_extractor, test_dl)
        train_preds = load_predictions(ckpt, x2c_extractor, train_dl)
        N_CONCEPTS  = test_preds["n_concepts"]
        EMB_SIZE    = test_preds["emb_size"]

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

        # --- CTL / ICL (share the flat c_mix) ---
        if todo["ctl"] or todo["icl_ksg"]:
            c_mix_flat = test_preds["c_mix"].reshape(len(test_preds["y_true"]), -1)

        if todo["ctl"]:
            print(f"\n  Computing CTL ({MI_REPEATS} repeats)...")
            ctl_mean, ctl_se = compute_ctl(c_mix_flat, test_preds["y_true"], N_CONCEPTS,
                                           repeats=MI_REPEATS, n_neighbors=N_NEIGHBORS)
            r["ctl_mean"] = ctl_mean.tolist()
            r["ctl_se"]   = ctl_se.tolist()
        print(f"  CTL per concept: {[f'{v:.4f}' for v in r['ctl_mean']]}  "
              f"(mean={np.mean(r['ctl_mean']):.4f})")

        if todo["icl_ksg"]:
            print(f"  Computing ICL ({MI_REPEATS} repeats)...")
            icl_mean, icl_se = compute_icl(c_mix_flat, N_CONCEPTS,
                                           repeats=MI_REPEATS, n_neighbors=N_NEIGHBORS)
            r["icl_mean"] = icl_mean
            r["icl_se"]   = icl_se
        print(f"  ICL={r['icl_mean']:.4f} ± {r['icl_se']:.4f}")

        # --- Probe accuracy vs PCA ---
        if todo["probe"]:
            print(f"  Computing probe accuracy curve (emb_size={EMB_SIZE} components)...")
            probe_acc = probe_accuracy_vs_pca(
                train_preds["c_mix"], test_preds["c_mix"],
                train_preds["y_true"], test_preds["y_true"],
                max_components=EMB_SIZE,
            )
            r["probe_acc"] = probe_acc.tolist()
        probe_acc = np.array(r["probe_acc"])
        print(f"  Probe acc at max components (per concept): "
              f"{[f'{probe_acc[k,-1]:.3f}' for k in range(N_CONCEPTS)]}")

        # --- CVL (Ridge-residual, legacy) ---
        if todo["cvl"]:
            print(f"  Computing CVL...")
            cvl = run_cvl(
                train_preds["c_mix"], test_preds["c_mix"],
                train_preds["c_true"], test_preds["c_true"],
                train_preds["y_true"], test_preds["y_true"],
            )
            r["cvl"] = cvl
        print(f"  CVL={r['cvl']['CVL']:.4f}  per-concept r2: "
              f"{[f'{v:.4f}' for v in r['cvl']['r2_per_concept']]}")

        # --- CVL global ---
        if todo["cvl_global"]:
            print(f"  Computing CVL_global...")
            r["cvl_global"] = run_cvl_global(
                train_preds["c_mix"], test_preds["c_mix"],
                train_preds["c_true"], test_preds["c_true"],
                train_preds["y_true"], test_preds["y_true"],
            )
        cg = r["cvl_global"]
        print(f"  CVL_global={cg['CVL_global']:.4f}  r2={cg.get('r2', cg.get('acc', 'n/a')):.4f}")

        # --- Embedding probe delta ---
        if todo["emb_delta"]:
            print(f"  Computing embedding probe delta (scalar vs embedding)...")
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
            print(f"  Computing inter-concept leakage probes...")
            r["icl_probe"] = interconcept_probe_analysis(
                train_preds["c_mix"], test_preds["c_mix"],
                train_preds["c_true"], test_preds["c_true"],
                baseline_cache_path=BASELINE_CACHE,
            )
        if "icl_probe" in r:
            print_icl_probe_summary(r["icl_probe"], label=f"{label} {fold}")

        # --- OIS / NIS / CAS ---
        if todo["ois_nis_cas"]:
            print(f"  Computing OIS / NIS / CAS...")
            r["ois_nis_cas"] = run_ois_nis_cas(ckpt, x2c_extractor, train_dl, test_dl)
        onc = r["ois_nis_cas"]
        print(f"  OIS={onc['ois']:.4f}  NIS={onc['nis']:.4f}  CAS={onc['cas']:.4f}")

        # --- ICVL ---
        if todo["icvl"]:
            print(f"  Computing ICVL ({N_CONCEPTS}×{N_CONCEPTS-1} logistic probes)...")
            r["icvl"] = run_icvl(
                train_preds["c_mix"], test_preds["c_mix"],
                train_preds["c_true"], test_preds["c_true"],
            )
        print(f"  ICVL={r['icvl']['ICVL']:.4f}")

        r["_complete"] = True
        results[label][LAM_C][fold] = r
        joblib.dump(results, SAVE_PATH)
        print(f"  Checkpoint saved → {SAVE_PATH}")

# ---------------------------------------------------------------------------
# PCA scatter and probe plots — use fold_1 of lam_c0.1 for visuals
# ---------------------------------------------------------------------------
_vis_lam = LAM_C_LIST[0]
_vis_cem_ckpts  = find_checkpoints(RESULTS_FOLDER, "CEM_",  _vis_lam, FOLDS)
_vis_acem_ckpts = find_checkpoints(RESULTS_FOLDER, "ACEM_", _vis_lam, FOLDS)

print("\nSaving PCA scatter plots...")
for label, fold_ckpts in [("cem", _vis_cem_ckpts), ("acem", _vis_acem_ckpts)]:
    first_fold = next(iter(fold_ckpts))
    test_preds = load_predictions(fold_ckpts[first_fold], x2c_extractor, test_dl)
    pca_scatter_concept(
        test_preds["c_mix"], test_preds["y_true"],
        concept_idx=PCA_SCATTER_CONCEPT,
        title=f"{label.upper()} — concept {PCA_SCATTER_CONCEPT} embeddings ({_vis_lam})",
        save_path=PLOT_DIR + f"pca_scatter_{label}_concept{PCA_SCATTER_CONCEPT}.png",
    )

print("Saving probe accuracy plots...")
first_fold = next(iter(_vis_cem_ckpts))
cem_probe  = np.array(results["cem"][_vis_lam][first_fold]["probe_acc"])
acem_probe = np.array(results["acem"][_vis_lam][first_fold]["probe_acc"])
for k in range(cem_probe.shape[0]):
    plot_probe_curve(
        cem_probe[k], acem_probe[k],
        title=f"TabularToy — concept {k} probe accuracy ({first_fold}, {_vis_lam})",
        save_path=PLOT_DIR + f"probe_curve_concept{k}.png",
        average=False,
    )
plot_probe_curve(
    cem_probe, acem_probe,
    title=f"TabularToy — probe accuracy avg over concepts ({first_fold}, {_vis_lam})",
    save_path=PLOT_DIR + "probe_curve_avg.png",
    average=True,
)

# ---------------------------------------------------------------------------
# Prototype alignment plot — fold_1 of lam_c0.1, CEM vs ACEM side-by-side
# ---------------------------------------------------------------------------
print("\nSaving prototype alignment plot...")
first_fold  = next(iter(_vis_cem_ckpts))
cem_preds   = load_predictions(_vis_cem_ckpts[first_fold],  x2c_extractor, test_dl)
acem_preds  = load_predictions(_vis_acem_ckpts[first_fold], x2c_extractor, test_dl)
proto_scores = prototype_alignment_plot(
    cem_preds["c_pos"],  cem_preds["c_neg"],
    acem_preds["c_pos"], acem_preds["c_neg"],
    label_a="CEM", label_b="ACEM",
    concept_names=[f"c{k}" for k in range(cem_preds["n_concepts"])],
    title=f"TabularToy — prototype alignment ({first_fold}, {_vis_lam})",
    save_path=PLOT_DIR + "prototype_alignment.png",
)
print(f"  CEM  pos-neg dist={proto_scores['CEM']['pos_neg_dist']:.4f}"
      f"  concept sep={proto_scores['CEM']['concept_sep']:.4f}")
print(f"  ACEM pos-neg dist={proto_scores['ACEM']['pos_neg_dist']:.4f}"
      f"  concept sep={proto_scores['ACEM']['concept_sep']:.4f}")

# ---------------------------------------------------------------------------
# Save and summary
# ---------------------------------------------------------------------------
joblib.dump(results, SAVE_PATH)
print(f"\nResults saved → {SAVE_PATH}")

print("\n=== Summary ===")
for label in ["cem", "acem"]:
    print(f"\n  {label.upper()}")
    for lam_c in LAM_C_LIST:
        if lam_c not in results.get(label, {}):
            continue
        print(f"  [{lam_c}]")
        for fold, r in results[label][lam_c].items():
            K = len(r["ctl_mean"])
            mask = ~np.eye(K, dtype=bool)
            n_wrong = sum(v["n_samples"] for k, v in r["error_set"].items() if k > 0)
            acc_wrong = np.mean([v["task_acc"] for k, v in r["error_set"].items() if k > 0])
            epd_str = (f"  EmbΔ={r['emb_probe_delta']['delta']:.4f}"
                       if "emb_probe_delta" in r else "")
            cvl_g_str = (f"  CVL_g={r['cvl_global']['CVL_global']:.4f}"
                         if "cvl_global" in r else "")
            onc_str = ""
            if "ois_nis_cas" in r:
                onc = r["ois_nis_cas"]
                onc_str = f"  OIS={onc['ois']:.4f}  NIS={onc['nis']:.4f}  CAS={onc['cas']:.4f}"
            icvl_str = f"  ICVL={r['icvl']['ICVL']:.4f}" if "icvl" in r else ""
            icl_probe_str = (f"  ICL_Δacc={r['icl_probe']['acc_diff'][mask].mean():.4f}"
                             if "icl_probe" in r else "")
            print(f"    {fold}  task={r['task_acc']:.4f}  CTL={np.mean(r['ctl_mean']):.4f}"
                  f"  ICL={r['icl_mean']:.4f}  CVL={r['cvl']['CVL']:.4f}{cvl_g_str}"
                  f"  ICVL={r['icvl']['ICVL']:.4f}"
                  f"  err_acc={acc_wrong:.4f}/{n_wrong}"
                  f"{icl_probe_str}{epd_str}{onc_str}")
