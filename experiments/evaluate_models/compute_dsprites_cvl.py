"""
Compute CVL for dSprites CEM and GC-CEM across all concept-loss weights.

Run with:
    conda run -n xai-leakage python experiments/evaluate_models/compute_dsprites_cvl.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)

import numpy as np
import joblib

from xai_concept_leakage.data.dsprites_auxiliary import dsprites_dataloaders
from xai_concept_leakage.metrics.leakage import compute_CVL
from experiments.experiment_utils import get_dsprites_extractor_arch
from experiments.evaluate_model import predict_c_y

SAVE_PATH = master + "results/results_dsprites_cvl.dict"

DS_CEM_FOLDER  = master + "results/dsprites_cem/"
DS_ACEM_FOLDER = master + "results/dsprites_acem_shared_critic/"

MODEL_SPECS = [
    ("cem",    "CEM",    DS_CEM_FOLDER,  "CEM_adam_lr1e-03_bs64_lam_c0.1",                           "lam_c0.1"),
    ("cem",    "CEM",    DS_CEM_FOLDER,  "CEM_adam_lr1e-03_bs64_lam_c0.5",                           "lam_c0.5"),
    ("cem",    "CEM",    DS_CEM_FOLDER,  "CEM_adam_lr1e-03_bs64_lam_c1",                             "lam_c1"),
    ("gc_cem", "GC-CEM", DS_ACEM_FOLDER, "CRCEM_adam_lr5e-05_bs64_lam1_none_lam_c0.1_shared_critic", "lam_c0.1"),
    ("gc_cem", "GC-CEM", DS_ACEM_FOLDER, "CRCEM_adam_lr5e-05_bs64_lam1_none_lam_c0.5_shared_critic", "lam_c0.5"),
    ("gc_cem", "GC-CEM", DS_ACEM_FOLDER, "CRCEM_adam_lr5e-05_bs64_lam1_none_lam_c1_shared_critic",   "lam_c1"),
]

ALL_FOLDS = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]

corr = 0
path_dataset = master + f"data/dsprites/dsprites_dep_{corr}.npz"
train_dl, val_dl, test_dl = dsprites_dataloaders(
    path_dataset, val_ratio=0.1, num_workers=0, batch_size=256
)
x2c_extractor = get_dsprites_extractor_arch


def _to_np(t):
    import torch
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def get_embeddings(ckpt_path):
    out_tr = predict_c_y(train_dl, ckpt_path, x2c_extractor,
                         c_sem_out=True, soft_prob_out=True, vec_emb_out=False)
    out_te = predict_c_y(test_dl,  ckpt_path, x2c_extractor,
                         c_sem_out=True, soft_prob_out=True, vec_emb_out=False)

    def unpack(out):
        c_sem, c_prob, c_true, y_pred, y_true = out[:5]
        c_true = _to_np(c_true)
        y_true = _to_np(y_true).ravel()
        n_k    = c_true.shape[1]
        emb    = _to_np(c_sem).reshape(-1, n_k, _to_np(c_sem).shape[1] // n_k)
        return emb, c_true, y_true

    return unpack(out_tr), unpack(out_te)


results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

for model_key, model_label, folder, model_name, lam in MODEL_SPECS:
    model_path = os.path.join(folder, model_name)
    results.setdefault(model_key, {}).setdefault(lam, {})
    print(f"\n{'='*60}")
    print(f"{model_label}  {lam}")

    for fold_key in ALL_FOLDS:
        ckpt = os.path.join(model_path, f"{model_name}_{fold_key}.pt")
        if not os.path.exists(ckpt):
            print(f"  {fold_key}: checkpoint missing")
            continue

        r = results[model_key][lam].get(fold_key, {})
        if "cvl_clipped_mean" in r:
            print(f"  {fold_key}: cached  CVL={r['cvl_clipped_mean']:.4f}")
            continue

        print(f"  {fold_key}: loading embeddings ...")
        (c_mix_tr, c_true_tr, y_tr), (c_mix_te, c_true_te, y_te) = get_embeddings(ckpt)
        print(f"    c_mix shape={c_mix_te.shape}  n_classes={len(np.unique(y_te))}")

        res = compute_CVL(c_mix_tr, c_mix_te, c_true_tr, c_true_te, y_tr, y_te)
        r["cvl_clipped_mean"]     = float(res["CVL"])
        r["cvl_clipped_per_concept"] = res["CVL_per_concept"]
        r["r2_per_concept"]       = res["r2_per_concept"]

        print(f"    CVL={r['cvl_clipped_mean']:.4f}")
        results[model_key][lam][fold_key] = r
        joblib.dump(results, SAVE_PATH)

print("\n\n=== SUMMARY ===")
print(f"{'Model':8} {'lam':10}  CVL")
for model_key, model_label, _, _, lam in MODEL_SPECS:
    folds = results.get(model_key, {}).get(lam, {})
    if not folds:
        continue
    vals = [r["cvl_clipped_mean"] for r in folds.values() if "cvl_clipped_mean" in r]
    if vals:
        print(f"{model_label:8} {lam:10}  {np.mean(vals):.4f}±{np.std(vals):.4f}")

print("\nDone.")
