"""
Step-1 explained variance (Ridge c_k -> ĉ_k) for CEM vs GC-CEM, across all
three datasets — extends check_step1_snr.py, which only covered TT CEM.

Motivation: on CUB (cub_cvl_pls.py), GC-CEM's step-1 explained variance
(45.5%) was ~8x higher than CEM's (5.9%) — i.e. the adversarial critic makes
each concept's embedding much more tightly determined by its own label. This
script checks whether that CEM-vs-GC-CEM gap is a general effect of the
adversarial critic or specific to CUB, by computing the same statistic for
TT and dSprites.

Embeddings for dSprites are cached to disk (results/step1_ev_dsprites_embeddings.joblib)
since the CNN forward pass takes a few minutes; TT is tabular/fast and not cached.

Run:
    conda run -n xai-leakage python experiments/evaluate_models/check_step1_explained_var.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)

import numpy as np
import joblib
from sklearn.linear_model import Ridge

from experiments.evaluate_model import predict_c_y


def step1_explained_var(emb_tr, c_tr):
    """Per-concept fraction of ĉ_k's variance explained by Ridge(c_k -> ĉ_k)."""
    K = emb_tr.shape[1]
    fracs = []
    for k in range(K):
        r = Ridge(alpha=1.0)
        r.fit(c_tr[:, k:k+1], emb_tr[:, k])
        R = emb_tr[:, k] - r.predict(c_tr[:, k:k+1])
        fracs.append(float(1.0 - R.var() / (emb_tr[:, k].var() + 1e-12)))
    return np.array(fracs)


def report(label, fracs):
    print(f"  {label:<10} mean={fracs.mean():.4f}  min={fracs.min():.4f}  "
          f"max={fracs.max():.4f}  (K={len(fracs)})")


# ---------------------------------------------------------------------------
# TabularToy: CEM vs GC-CEM
# ---------------------------------------------------------------------------
print("=" * 70)
print("TabularToy  lam_c0.1  fold_1")
print("=" * 70)

from xai_concept_leakage.data.tabulartoy_auxiliary import TT_dataloaders

tt_save_folder = master + "data/TabularToy/tabulartoy_25_10k/"
tt_train_dl, _, tt_test_dl = TT_dataloaders(
    tt_save_folder, considered_concepts=["0", "1", "2"], c_logits=False, num_workers=0)

from experiments.experiment_utils import get_tabulartoy_extractor_arch


def unpack_tt(out, n_k):
    c_sem, c_prob, c_true, y_pred, y_true = out[:5]
    c_sem  = c_sem.detach().cpu().numpy()
    c_true = c_true.detach().cpu().numpy()
    emb = c_sem.reshape(-1, n_k, c_sem.shape[1] // n_k)
    return emb, c_true


TT_MODELS = [
    ("CEM",
     "results/tabulartoy_25_10k_models_acem_shared_critic/"
     "CEM_adam_lr0.05_bs64_lam_c0.1/CEM_adam_lr0.05_bs64_lam_c0.1_fold_1.pt"),
    ("GC-CEM",
     "results/tabulartoy_25_10k_models_acem_shared_critic/"
     "ACEM_adam_lr1e-03_bs64_lam1_none_lam_c0.1_shared_critic/"
     "ACEM_adam_lr1e-03_bs64_lam1_none_lam_c0.1_shared_critic_fold_1.pt"),
]

for label, rel_ckpt in TT_MODELS:
    ckpt = master + rel_ckpt
    if not os.path.exists(ckpt):
        print(f"  {label}: checkpoint missing — {ckpt}")
        continue
    out_tr = predict_c_y(tt_train_dl, ckpt, get_tabulartoy_extractor_arch,
                         c_sem_out=True, soft_prob_out=True, vec_emb_out=False)
    emb_tr, c_tr = unpack_tt(out_tr, 3)
    fracs = step1_explained_var(emb_tr, c_tr)
    report(label, fracs)

# ---------------------------------------------------------------------------
# dSprites: CEM vs GC-CEM
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("dSprites  lam_c0.1  fold_1")
print("=" * 70)

from xai_concept_leakage.data.dsprites_auxiliary import dsprites_dataloaders
from experiments.experiment_utils import get_dsprites_extractor_arch

ds_train_dl, _, ds_test_dl = dsprites_dataloaders(
    master + "data/dsprites/dsprites_dep_0.npz", val_ratio=0.1, num_workers=0, batch_size=256)


def unpack_ds(out, n_k):
    c_sem, c_prob, c_true, y_pred, y_true = out[:5]
    c_sem  = c_sem.detach().cpu().numpy()
    c_true = c_true.detach().cpu().numpy()
    emb = c_sem.reshape(-1, n_k, c_sem.shape[1] // n_k)
    return emb, c_true


DS_MODELS = [
    ("CEM",
     "results/dsprites_cem/CEM_adam_lr1e-03_bs64_lam_c0.1/"
     "CEM_adam_lr1e-03_bs64_lam_c0.1_fold_1.pt"),
    ("GC-CEM",
     "results/dsprites_acem_shared_critic/"
     "CRCEM_adam_lr5e-05_bs64_lam1_none_lam_c0.1_shared_critic/"
     "CRCEM_adam_lr5e-05_bs64_lam1_none_lam_c0.1_shared_critic_fold_1.pt"),
]

DS_CACHE_PATH = master + "results/step1_ev_dsprites_embeddings.joblib"
ds_cache = joblib.load(DS_CACHE_PATH) if os.path.exists(DS_CACHE_PATH) else {}

for label, rel_ckpt in DS_MODELS:
    ckpt = master + rel_ckpt
    if not os.path.exists(ckpt):
        print(f"  {label}: checkpoint missing — {ckpt}")
        continue
    if label in ds_cache:
        emb_tr, c_tr = ds_cache[label]
        print(f"  {label}: loaded embeddings from cache")
    else:
        print(f"  {label}: running inference...")
        out_tr = predict_c_y(ds_train_dl, ckpt, get_dsprites_extractor_arch,
                             c_sem_out=True, soft_prob_out=True, vec_emb_out=False)
        emb_tr, c_tr = unpack_ds(out_tr, 5)
        ds_cache[label] = (emb_tr, c_tr)
        joblib.dump(ds_cache, DS_CACHE_PATH)
        print(f"  {label}: saved embeddings → {DS_CACHE_PATH}")
    fracs = step1_explained_var(emb_tr, c_tr)
    report(label, fracs)

# ---------------------------------------------------------------------------
# CUB: already computed by cub_cvl_pls.py — restated here for comparison
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("CUB  lam_c0.1  fold_1  (from cub_cvl_pls.py run)")
print("=" * 70)
print(f"  {'CEM':<10} mean=0.0593  min=0.0320  max=0.1195  (K=112)")
print(f"  {'GC-CEM':<10} mean=0.4545  min=0.0905  max=0.8631  (K=112)")

print("\nDone.")
