"""
CEM/CRCEM evaluation suite for CelebA (39 concepts, 2 classes: Male/not-Male).

Metrics
-------
- Task accuracy, per-concept accuracy
- Error-set analysis (task acc bucketed by # wrong concepts)
- Probe accuracy vs PCA components (averaged across all concepts)
- CVL global (single regression over full [N, K*m] embedding)
- Per-concept CVL
- Embedding probe delta (scalar vs embedding)
- Task leakage probe (y|c vs y|[c, Ĉ])
- Intervention curve (random policy, group-level)
- Adversarial concept flip
- Inter-concept leakage probe (K×K acc matrix)
- ICVL (inter-concept variance leakage)
- OIS / NIS / CAS

KSG CTL/ICL are skipped — 39 concepts × 16-dim × large dataset makes KSG infeasibly slow.

Folder layout
-------------
lam_c0.1  → results/celeba_cem_39c/
lam_c1    → results/celeba_cem_39c_lam1_0/   (CRCEM only; no CEM lam_c1 config)

Results saved to  results/results_celeba_suite.dict
Plots saved to    results/plots/celeba/
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

master_folder = os.getcwd().replace("/experiments/evaluate_models", "")
sys.path.insert(0, master_folder)
master_folder = master_folder + "/"

import numpy as np
import joblib
import xai_concept_leakage.data.celeba_loader as celeba_data_module


import torch.utils.data as tud


class CelebADatasetWrapper(tud.Dataset):
    """Wraps a CelebA dataset so each sample returns (x, y, c) instead of (x, (y, c))."""
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        x, (y, c) = self.ds[idx]
        return x, y, c


def wrap_celeba_dl(dl):
    """Return a new DataLoader whose dataset yields (x, y, c) tuples."""
    wrapped_ds = CelebADatasetWrapper(dl.dataset)
    return tud.DataLoader(
        wrapped_ds,
        batch_size=dl.batch_size,
        shuffle=isinstance(dl.sampler, tud.RandomSampler),
        num_workers=dl.num_workers,
        pin_memory=dl.pin_memory,
    )

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
TRIAL_RUN        = False
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
RERUN_OIS_NIS_CAS = False
RERUN_ICVL        = False
RUN_OIS_NIS_CAS   = False   # requires tensorflow — skip locally
RUN_ICL_PROBE     = True
RUN_ICVL          = True
INTERVENTION_POLICIES = ["random"]
INTERVENTION_REPEATS  = 3

# Folder that holds the results for each lam_c value
LAM_FOLDERS = {
    "lam_c0.1": master_folder + "results/celeba_cem_39c/",
    "lam_c1":   master_folder + "results/celeba_cem_39c_lam1_0/",
}
LAM_C_LIST     = list(LAM_FOLDERS.keys())
FOLDS          = ["fold_1"] if TRIAL_RUN else ALL_FOLDS
MAX_PCA_COMP   = 32

SAVE_PATH      = master_folder + "results/results_celeba_suite.dict"
PLOT_DIR       = master_folder + "results/plots/celeba/"
BASELINE_CACHE = master_folder + "results/cache/celeba_icl_baseline.joblib"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(master_folder + "results/cache/", exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
dataset_config = {
    "dataset": "celeba",
    "num_workers": 0,
    "batch_size": 256,
    "root_dir": master_folder + "data/",
    "label_attr_idx": 20,      # Male attribute (index 20)
    "num_concepts": 39,         # all attributes except Male
    "num_classes": 2,
    "image_size": 64,
    "sampling_percent": 1,
    "test_subsampling": 1,
    "weight_loss": False,
    "train_augment": False,
}
train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = (
    celeba_data_module.generate_data(
        config=dataset_config, seed=42,
        output_dataset_vars=True,
        root_dir=dataset_config["root_dir"],
    )
)
train_dl = wrap_celeba_dl(train_dl)
val_dl   = wrap_celeba_dl(val_dl)
test_dl  = wrap_celeba_dl(test_dl)
x2c_extractor = None
print(f"CelebA: n_concepts={n_concepts}, n_tasks={n_tasks}")

# ---------------------------------------------------------------------------
# Metrics — iterate over lambda, model type, fold
# ---------------------------------------------------------------------------
results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

for LAM_C in LAM_C_LIST:
    folder = LAM_FOLDERS[LAM_C]
    if not os.path.isdir(folder):
        print(f"\n  Skipping {LAM_C} — folder not found: {folder}")
        continue

    cem_ckpts   = find_checkpoints(folder, "CEM_",   LAM_C, FOLDS)
    crcem_ckpts = find_checkpoints(folder, "CRCEM_", LAM_C, FOLDS)

    print(f"\n{'#'*60}")
    print(f"  LAM_C={LAM_C}  folder={os.path.basename(folder.rstrip('/'))}")
    print(f"{'#'*60}")
    for fold, path in cem_ckpts.items():
        print(f"  CEM   {fold}: {os.path.relpath(path, master_folder)}")
    for fold, path in crcem_ckpts.items():
        print(f"  CRCEM {fold}: {os.path.relpath(path, master_folder)}")

    for label, fold_ckpts in [("cem", cem_ckpts), ("crcem", crcem_ckpts)]:
        if not fold_ckpts:
            print(f"  No {label.upper()} checkpoints found for {LAM_C}, skipping.")
            continue
        results.setdefault(label, {}).setdefault(LAM_C, {})

        for fold, ckpt in fold_ckpts.items():
            r = results[label][LAM_C].get(fold, {})

            todo = {
                "task":        "task_acc"        not in r or RERUN_TASK,
                "errset":      "error_set"        not in r or RERUN_ERRSET,
                "probe":       "probe_acc"        not in r or RERUN_PROBE,
                "cvl_global":  "cvl_global"       not in r or RERUN_CVL_GLOBAL,
                "cvl":         "cvl"              not in r or RERUN_CVL,
                "emb_delta":   "emb_probe_delta"  not in r or RERUN_EMB_DELTA,
                "task_leak":   "task_leak"        not in r or RERUN_TASK_LEAK,
                "interv":      "interv"           not in r or RERUN_INTERV,
                "adv_flip":    "adv_flip"         not in r or RERUN_ADV_FLIP,
                "icl_probe":   ("icl_probe" not in r and RUN_ICL_PROBE) or RERUN_ICL_PROBE,
                "ois_nis_cas": ("ois_nis_cas" not in r and RUN_OIS_NIS_CAS) or RERUN_OIS_NIS_CAS,
                "icvl":        ("icvl" not in r and RUN_ICVL) or RERUN_ICVL,
            }

            if not any(todo.values()):
                print(f"\n  Skipping {label.upper()} {fold} ({LAM_C}) — all metrics cached")
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
            max_comp   = min(MAX_PCA_COMP, EMB_SIZE)

            # --- task accuracy ---
            if todo["task"]:
                r["task_acc"] = float((test_preds["y_pred"].argmax(-1) == test_preds["y_true"]).mean())
                r["c_acc"]    = float((test_preds["c_pred"] == test_preds["c_true"]).mean())
            print(f"  task_acc={r['task_acc']:.4f}  c_acc={r['c_acc']:.4f}")

            # --- error-set analysis ---
            if todo["errset"]:
                r["error_set"] = error_set_analysis(
                    test_preds["c_pred"], test_preds["c_true"],
                    test_preds["y_pred"], test_preds["y_true"],
                )
            print_error_set(r["error_set"], label=f"{label} {fold}")

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
            print(f"  Probe acc at {max_comp} components: {probe_acc[:, -1].mean():.4f}")

            # --- CVL global ---
            if todo["cvl_global"]:
                print(f"  Computing CVL_global ({N_CONCEPTS} × {EMB_SIZE}-dim)...")
                r["cvl_global"] = run_cvl_global(
                    train_preds["c_mix"], test_preds["c_mix"],
                    train_preds["c_true"], test_preds["c_true"],
                    train_preds["y_true"], test_preds["y_true"],
                )
            cg = r["cvl_global"]
            print(f"  CVL_global={cg['CVL_global']:.4f}")

            # --- Per-concept CVL ---
            if todo["cvl"]:
                print(f"  Computing per-concept CVL ({N_CONCEPTS} × {EMB_SIZE}-dim)...")
                r["cvl"] = run_cvl(
                    train_preds["c_mix"], test_preds["c_mix"],
                    train_preds["c_true"], test_preds["c_true"],
                    train_preds["y_true"], test_preds["y_true"],
                )
            cv = r["cvl"]
            print(f"  CVL={cv['CVL']:.4f}")

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

            # --- Adversarial concept flip ---
            if todo["adv_flip"]:
                print(f"  Computing adversarial concept flip...")
                r["adv_flip"] = adversarial_concept_flip(
                    ckpt, x2c_extractor, train_dl, test_preds
                )
            af = r["adv_flip"]
            print(f"  Adv flip: 0→{af['acc_0']:.4f}  all→{af['acc_all']:.4f}  drop={af['acc_0']-af['acc_all']:.4f}")

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

            # --- ICVL ---
            if todo["icvl"]:
                print(f"  Computing ICVL ({N_CONCEPTS}×{N_CONCEPTS-1} logistic probes)...")
                r["icvl"] = run_icvl(
                    train_preds["c_mix"], test_preds["c_mix"],
                    train_preds["c_true"], test_preds["c_true"],
                )
            if "icvl" in r:
                print(f"  ICVL={r['icvl']['ICVL']:.4f}")

            # OIS/NIS/CAS skipped — requires tensorflow subprocess

            r["_complete"] = True
            results[label][LAM_C][fold] = r
            joblib.dump(results, SAVE_PATH)
            print(f"  Checkpoint saved → {SAVE_PATH}")

# ---------------------------------------------------------------------------
# Probe plot
# ---------------------------------------------------------------------------
print("\nSaving probe accuracy plot...")
_vis_lam   = "lam_c0.1"
_vis_folder = LAM_FOLDERS[_vis_lam]
if os.path.isdir(_vis_folder):
    _cem_ckpts   = find_checkpoints(_vis_folder, "CEM_",   _vis_lam, FOLDS)
    _crcem_ckpts = find_checkpoints(_vis_folder, "CRCEM_", _vis_lam, FOLDS)
    if (_vis_lam in results.get("cem", {}) and _cem_ckpts
            and _vis_lam in results.get("crcem", {}) and _crcem_ckpts):
        first_fold    = next(iter(_cem_ckpts))
        cem_probe     = np.array(results["cem"][_vis_lam][first_fold]["probe_acc"])
        crcem_probe   = np.array(results["crcem"][_vis_lam][first_fold]["probe_acc"])
        plot_probe_curve(
            cem_probe, crcem_probe,
            title=f"CelebA — probe accuracy (avg over {n_concepts} concepts, {first_fold})",
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
            line = f"      {fold}  task={r.get('task_acc', float('nan')):.4f}"
            if "probe_acc" in r:
                probe = np.array(r["probe_acc"])
                line += f"  probe@{len(r['probe_acc'][0])}={probe[:,-1].mean():.4f}"
            if "cvl_global" in r:
                line += f"  CVL_g={r['cvl_global']['CVL_global']:.4f}"
            if "cvl" in r:
                line += f"  CVL={r['cvl']['CVL']:.4f}"
            if "emb_probe_delta" in r:
                line += f"  EmbΔ={r['emb_probe_delta']['delta']:.4f}"
            if "icvl" in r:
                line += f"  ICVL={r['icvl']['ICVL']:.4f}"
            print(line)
