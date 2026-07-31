"""
Compute CVL for TabularToy CEM and GC-CEM across all concept-loss weights.

Run with:
    conda run -n xai-leakage python experiments/evaluate_models/compute_tabulartoy_cvl.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)

import numpy as np
import joblib

from xai_concept_leakage.data.tabulartoy_auxiliary import TT_dataloaders
from xai_concept_leakage.metrics.leakage import compute_CVL
from experiments.experiment_utils import get_tabulartoy_extractor_arch
from experiments.evaluate_model import predict_c_y

SAVE_PATH = master + "results/results_tabulartoy_cvl.dict"

TT_ACEM_FOLDER = master + "results/tabulartoy_25_10k_models_acem_shared_critic/"

MODEL_SPECS = [
    ("cem",    "CEM",    TT_ACEM_FOLDER, "CEM_adam_lr0.05_bs64_lam_c0.1",                           "lam_c0.1"),
    ("cem",    "CEM",    TT_ACEM_FOLDER, "CEM_adam_lr0.05_bs64_lam_c0.5",                           "lam_c0.5"),
    ("cem",    "CEM",    TT_ACEM_FOLDER, "CEM_adam_lr0.05_bs64_lam_c1",                             "lam_c1"),
    ("gc_cem", "GC-CEM", TT_ACEM_FOLDER, "ACEM_adam_lr1e-03_bs64_lam1_none_lam_c0.1_shared_critic", "lam_c0.1"),
    ("gc_cem", "GC-CEM", TT_ACEM_FOLDER, "ACEM_adam_lr1e-03_bs64_lam1_none_lam_c0.5_shared_critic", "lam_c0.5"),
    ("gc_cem", "GC-CEM", TT_ACEM_FOLDER, "ACEM_adam_lr1e-03_bs64_lam1_none_lam_c1_shared_critic",   "lam_c1"),
]

ALL_FOLDS = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]

save_folder = master + "data/TabularToy/tabulartoy_25_10k/"
train_dl, val_dl, test_dl = TT_dataloaders(
    save_folder, considered_concepts=["0", "1", "2"], c_logits=False, num_workers=0
)
x2c_extractor = get_tabulartoy_extractor_arch


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
    if not os.path.exists(model_path):
        print(f"MISSING: {model_path}")
        continue
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
        (emb_tr, c_tr, y_tr), (emb_te, c_te, y_te) = get_embeddings(ckpt)
        print(f"    emb shape={emb_te.shape}  n_classes={len(np.unique(y_te))}")

        res = compute_CVL(emb_tr, emb_te, c_tr, c_te, y_tr, y_te)
        r["cvl_clipped_mean"]        = float(res["CVL"])
        r["cvl_clipped_per_concept"] = res["CVL_per_concept"]
        r["r2_per_concept"]          = res["r2_per_concept"]

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
