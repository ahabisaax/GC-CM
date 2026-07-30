"""
Compute CVL and ICVL for all CUB CEM/GC-CEM models across all folds.

Reports:
  - CVL per concept (raw R² before clipping + clipped)
  - CVL global (raw + clipped)
  - ICVL (raw diagonal-excluded R² mean + clipped)

Results cached in results/results_cub_cvl_icvl.dict

Run with:
    conda run -n xai-leakage python experiments/evaluate_models/compute_cub_cvl_icvl.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)
sys.path.insert(0, master + "data/CUB200")

import numpy as np
import joblib
import data.CUB200.cub_loader as cub_data_module

from experiments.evaluate_model import predict_c_y
from xai_concept_leakage.metrics.leakage import (
    compute_CVL, compute_CVL_global, compute_ICVL,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RERUN = False   # set True to recompute even if cached

CUB_CEM_FOLDER  = master + "results/cub_cem/"
SAVE_PATH       = master + "results/results_cub_cvl_icvl.dict"

MODEL_SPECS = [
    ("cem",    "CEM",    "CEM_adam_lr1e-03_bs256_lam_c0.1",                          "lam_c0.1"),
    ("cem",    "CEM",    "CEM_adam_lr1e-03_bs256_lam_c0.5",                          "lam_c0.5"),
    ("cem",    "CEM",    "CEM_adam_lr1e-03_bs256_lam_c1",                            "lam_c1"),
    ("gc_cem", "GC-CEM", "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c0.1_shared_critic","lam_c0.1"),
    ("gc_cem", "GC-CEM", "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c0.5_shared_critic","lam_c0.5"),
    ("gc_cem", "GC-CEM", "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c1_shared_critic",  "lam_c1"),
]

ALL_FOLDS = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
dataset_config = {
    "dataset": "cub",
    "root_dir": master + "data/CUB200/",
    "batch_size": 256,
    "num_workers": 0,
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
print(f"CUB: n_concepts={n_concepts}, n_tasks={n_tasks}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_np(t):
    import torch
    if isinstance(t, np.ndarray):
        return t
    return t.detach().cpu().numpy()


def get_embeddings(ckpt_path):
    """Run inference and return c_mix [N, K, m], c_true [N, K], y_true [N]."""
    out_train = predict_c_y(
        dl=train_dl, model_path=ckpt_path, x2c_extractor=None,
        c_sem_out=True, soft_prob_out=True, vec_emb_out=False,
    )
    out_test = predict_c_y(
        dl=test_dl,  model_path=ckpt_path, x2c_extractor=None,
        c_sem_out=True, soft_prob_out=True, vec_emb_out=False,
    )

    def unpack(out):
        c_sem, c_prob, c_true, y_pred, y_true = out[:5]
        c_prob  = _to_np(c_prob)
        c_true  = _to_np(c_true)
        y_true  = _to_np(y_true).ravel()
        n_k     = c_true.shape[1]
        emb     = _to_np(c_sem).reshape(-1, n_k, _to_np(c_sem).shape[1] // n_k)
        return emb, c_true, y_true

    return unpack(out_train), unpack(out_test)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

for model_key, model_label, model_name, lam in MODEL_SPECS:
    model_path = os.path.join(CUB_CEM_FOLDER, model_name)
    results.setdefault(model_key, {}).setdefault(lam, {})
    print(f"\n{'='*60}")
    print(f"{model_label}  {lam}  ({model_name})")

    for fold_key in ALL_FOLDS:
        fold_1idx = int(fold_key.split("_")[1])
        fold_0idx = fold_1idx - 1
        ckpt = os.path.join(model_path, f"{model_name}_{fold_key}.pt")
        if not os.path.exists(ckpt):
            print(f"  {fold_key}: checkpoint missing")
            continue

        r = results[model_key][lam].get(fold_key, {})
        need_cvl  = "cvl_raw"  not in r or RERUN
        need_icvl = "icvl_raw" not in r or RERUN

        if not (need_cvl or need_icvl):
            print(f"  {fold_key}: cached (CVL={r['cvl_raw_mean']:.4f}  ICVL={r['icvl_raw_mean']:.4f})")
            continue

        print(f"  {fold_key}: loading embeddings …")
        (c_mix_tr, c_true_tr, y_tr), (c_mix_te, c_true_te, y_te) = get_embeddings(ckpt)
        print(f"    c_mix shape: {c_mix_te.shape}  y unique: {len(np.unique(y_te))}")

        if need_cvl:
            print(f"    computing CVL (per-concept) …")
            cvl_result = compute_CVL(c_mix_tr, c_mix_te, c_true_tr, c_true_te, y_tr, y_te)
            r2_raw = np.array(cvl_result["r2_per_concept"])
            r["cvl_raw"]      = cvl_result["r2_per_concept"]       # raw before clip
            r["cvl_clipped"]  = cvl_result["CVL_per_concept"]      # after max(0,.)
            r["cvl_raw_mean"] = float(np.mean(r2_raw))
            r["cvl_clipped_mean"] = float(cvl_result["CVL"])
            print(f"    CVL raw r2: min={r2_raw.min():.4f}  max={r2_raw.max():.4f}  mean={r2_raw.mean():.4f}")
            print(f"    CVL clipped mean={cvl_result['CVL']:.4f}")

            print(f"    computing CVL_global …")
            cvlg = compute_CVL_global(c_mix_tr, c_mix_te, c_true_tr, c_true_te, y_tr, y_te)
            r["cvl_global_raw"]     = float(cvlg.get("r2", float("nan")))
            r["cvl_global_clipped"] = float(cvlg.get("CVL_global", 0.0))
            print(f"    CVL_global raw r2={r['cvl_global_raw']:.4f}  clipped={r['cvl_global_clipped']:.4f}")

        if need_icvl:
            print(f"    computing ICVL …")
            icvl_result = compute_ICVL(c_mix_tr, c_mix_te, c_true_tr, c_true_te)
            r2_mat = np.array(icvl_result["r2_matrix"])
            # off-diagonal raw values
            mask = ~np.eye(r2_mat.shape[0], dtype=bool)
            r2_offdiag = r2_mat[mask]
            r["icvl_raw_mean"]     = float(np.mean(r2_offdiag))
            r["icvl_raw_max"]      = float(np.max(r2_offdiag))
            r["icvl_clipped"]      = float(icvl_result["ICVL"])
            print(f"    ICVL raw off-diag: min={r2_offdiag.min():.4f}  max={r2_offdiag.max():.4f}  mean={r2_offdiag.mean():.4f}")
            print(f"    ICVL clipped={icvl_result['ICVL']:.6f}")

        results[model_key][lam][fold_key] = r
        joblib.dump(results, SAVE_PATH)
        print(f"    Saved → {SAVE_PATH}")

print("\n\n=== SUMMARY ===")
for model_key, model_label, model_name, lam in MODEL_SPECS:
    folds = results.get(model_key, {}).get(lam, {})
    if not folds:
        continue
    cvl_raws    = [r["cvl_raw_mean"]      for r in folds.values() if "cvl_raw_mean"  in r]
    cvl_clips   = [r["cvl_clipped_mean"]  for r in folds.values() if "cvl_clipped_mean" in r]
    cvlg_raws   = [r["cvl_global_raw"]    for r in folds.values() if "cvl_global_raw"   in r]
    icvl_raws   = [r["icvl_raw_mean"]     for r in folds.values() if "icvl_raw_mean"    in r]
    icvl_clips  = [r["icvl_clipped"]      for r in folds.values() if "icvl_clipped"     in r]
    print(f"{model_label:8} {lam:10}  "
          f"CVL_raw={np.mean(cvl_raws):.4f}  CVL={np.mean(cvl_clips):.4f}  "
          f"CVL_global_raw={np.mean(cvlg_raws):.4f}  "
          f"ICVL_raw={np.mean(icvl_raws):.4f}  ICVL={np.mean(icvl_clips):.6f}")

print("\nDone.")
