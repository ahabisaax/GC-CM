"""
PLS-based CVL diagnostic on CUB — one fold each for CEM and GC-CEM.

Ridge's step-2 regression (Y_onehot -> R_k) fits all directions of the
residual embedding, penalised by magnitude only — it has no notion of which
directions actually covary with the task label. When true leakage is a
small fraction of R_k's variance (as on CUB, where step 1 already explains
most of ĉ_k's variance from the concept's own label, and Y is a 200-column
one-hot), Ridge spreads its fit across mostly-noise directions and
out-of-sample R² washes into slightly-negative noise — which then clips to
a flat 0 for every concept.

PLSRegression replaces Ridge in step 2: it finds the directions of R_k that
maximally covary with Y and regresses only in that subspace — a targeted
search for the leakage direction instead of a magnitude-penalised fit over
everything. See xai_concept_leakage/metrics/leakage.py::compute_CVL_pls /
compute_CVL_global_pls.

Sweeps n_components for both per-concept and global CVL, and reports the
step-1 explained variance (how much of ĉ_k's variance the concept's own
label already accounts for) once per model, so we know how expressive/lossy
step 1 is before the leakage probe even runs.

Also decomposes the global raw R² into its numerator (residual MSE after
the step-2 fit) and denominator (test residual variance after step 1) for
both models, since GC-CEM's raw R² comes out ~5x more negative than CEM's —
this shows whether that gap is a scale artifact of step 1 already removing
much more variance for GC-CEM (leaving a small, noise-dominated residual to
fit in step 2) or something else.

Embeddings are cached to disk (results/cub_cvl_pls_embeddings_fold1_lamc0.1.joblib)
after the first CPU inference pass, since re-running the CNN forward pass on
the full CUB train+test sets takes ~25 min per model on CPU (no GPU/MPS is
used for this feature-extraction step).

Run:
    conda run -n xai-leakage python experiments/evaluate_models/cub_cvl_pls.py
"""
import os, sys, warnings, contextlib, io
warnings.filterwarnings("ignore")
master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)
sys.path.insert(0, master + "data/CUB200")

import numpy as np
import joblib
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
import data.CUB200.cub_loader as cub_data_module
from experiments.evaluate_model import predict_c_y
from xai_concept_leakage.metrics.leakage import compute_CVL_pls, compute_CVL_global_pls

N_COMPONENTS_SWEEP = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]

MODELS = [
    ("CEM",
     "results/cub_cem/CEM_adam_lr1e-03_bs256_lam_c0.1/"
     "CEM_adam_lr1e-03_bs256_lam_c0.1_fold_1.pt"),
    ("GC-CEM",
     "results/cub_cem/CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c0.1_shared_critic/"
     "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c0.1_shared_critic_fold_1.pt"),
]

CACHE_PATH = master + "results/cub_cvl_pls_embeddings_fold1_lamc0.1.joblib"
_cache = joblib.load(CACHE_PATH) if os.path.exists(CACHE_PATH) else {}

# Skip CUB dataset construction entirely when every model is already cached —
# generate_data() itself takes several minutes even though no CNN inference
# runs on a cache hit.
train_dl = test_dl = None
n_concepts = n_tasks = None
if not all(label in _cache for label, _ in MODELS):
    dataset_config = {
        "dataset": "cub", "root_dir": master + "data/CUB200/",
        "batch_size": 256, "num_workers": 0, "sampling_percent": 1,
        "sampling_groups": True, "test_subsampling": 1,
        "weight_loss": True, "train_augment": False,
    }
    train_dl, _, test_dl, _, (n_concepts, n_tasks, _) = cub_data_module.generate_data(
        config=dataset_config, seed=42, output_dataset_vars=True,
        root_dir=dataset_config["root_dir"])
    print(f"n_concepts={n_concepts}, n_tasks={n_tasks}")
else:
    print("All models cached — skipping CUB dataset construction.")


def unpack(out):
    c_sem, c_prob, c_true, y_pred, y_true = out[:5]
    c_sem  = c_sem.detach().cpu().numpy()
    c_true = c_true.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy().ravel()
    emb = c_sem.reshape(-1, n_concepts, c_sem.shape[1] // n_concepts)
    return emb, c_true, y_true


def get_embeddings(label, ckpt):
    if label in _cache:
        print(f"  {label}: loaded embeddings from cache ({CACHE_PATH})")
        return _cache[label]
    print("  Loading train embeddings...")
    emb_tr, c_tr, y_tr = unpack(predict_c_y(
        train_dl, ckpt, None, c_sem_out=True, soft_prob_out=True, vec_emb_out=False))
    print("  Loading test embeddings...")
    emb_te, c_te, y_te = unpack(predict_c_y(
        test_dl,  ckpt, None, c_sem_out=True, soft_prob_out=True, vec_emb_out=False))
    _cache[label] = (emb_tr, c_tr, y_tr, emb_te, c_te, y_te)
    joblib.dump(_cache, CACHE_PATH)
    print(f"  Saved embeddings → {CACHE_PATH}")
    return _cache[label]


for label, rel_ckpt in MODELS:
    ckpt = master + rel_ckpt
    if not os.path.exists(ckpt):
        print(f"\n{label}: checkpoint missing — {ckpt}")
        continue

    print(f"\n{'='*70}\n{label}  lam_c0.1  fold_1\n{'='*70}")
    emb_tr, c_tr, y_tr, emb_te, c_te, y_te = get_embeddings(label, ckpt)
    print(f"  emb shape: {emb_tr.shape}  |  {len(np.unique(y_tr))} classes  "
          f"|  N_test={len(y_te)}")

    # Step 1 (Ridge c_k -> ĉ_k) doesn't depend on n_components — probe once.
    with contextlib.redirect_stdout(io.StringIO()):
        per_probe  = compute_CVL_pls(emb_tr, emb_te, c_tr, c_te, y_tr, y_te,
                                     n_components=1)
        glob_probe = compute_CVL_global_pls(emb_tr, emb_te, c_tr, c_te, y_tr, y_te,
                                            n_components=1)
    ev = np.array(per_probe["step1_explained_var_per_concept"])
    ev_g = np.array(glob_probe["step1_explained_var_per_concept"])
    print(f"\n  Step-1 explained variance of ĉ_k by c_k's own label (Ridge):")
    print(f"    per-concept fit   : mean={ev.mean():.4f}  min={ev.min():.4f}  max={ev.max():.4f}")
    print(f"    global joint fit  : mean={ev_g.mean():.4f}  min={ev_g.min():.4f}  max={ev_g.max():.4f}"
          f"  (overall={glob_probe['step1_explained_var']:.4f})")
    print(f"    -> ~{ev.mean()*100:.1f}% of each ĉ_k's variance is already explained "
          f"by its own label; leakage can only live in the remaining "
          f"~{(1-ev.mean())*100:.1f}%.")

    # Decompose global raw R² = 1 - MSE_pred/MSE_baseline at n_comp=1, to see
    # whether GC-CEM's much more negative R² is a scale artifact of step 1
    # already removing far more variance (small residual, noise-dominated)
    # rather than a real difference in overfitting behaviour.
    from sklearn.preprocessing import label_binarize
    classes = np.unique(y_tr)
    Y_tr_ = label_binarize(y_tr, classes=classes).astype(float)
    Y_te_ = label_binarize(y_te, classes=classes).astype(float)
    N_tr, K, m = emb_tr.shape
    N_te = emb_te.shape[0]
    X_tr = emb_tr.reshape(N_tr, K * m)
    X_te = emb_te.reshape(N_te, K * m)
    r1 = Ridge(alpha=1.0)
    r1.fit(c_tr, X_tr)
    R_tr = X_tr - r1.predict(c_tr)
    R_te = X_te - r1.predict(c_te)
    pls1 = PLSRegression(n_components=1)
    pls1.fit(Y_tr_, R_tr)
    pred_te = pls1.predict(Y_te_)
    mse_pred = float(np.mean((R_te - pred_te) ** 2))
    mse_baseline = float(R_te.var())
    print(f"\n  R² decomposition (global, n_comp=1): "
          f"N_train={N_tr}  N_test={N_te}  Y_dims={Y_tr_.shape[1]}  R_dims={K*m}")
    print(f"    MSE_baseline (test residual var) = {mse_baseline:.6f}")
    print(f"    MSE_pred     (after PLS fit)     = {mse_pred:.6f}")
    print(f"    R² = 1 - MSE_pred/MSE_baseline   = {1 - mse_pred/mse_baseline:.4f}")

    print(f"\n  {'n_comp':>7}  {'per_concept_CVL':>16}  {'global_CVL':>12}  "
          f"{'global_r2_raw':>13}")
    print("  " + "-" * 55)
    for n_comp in N_COMPONENTS_SWEEP:
        with contextlib.redirect_stdout(io.StringIO()):
            per  = compute_CVL_pls(emb_tr, emb_te, c_tr, c_te, y_tr, y_te,
                                   n_components=n_comp)
            glob = compute_CVL_global_pls(emb_tr, emb_te, c_tr, c_te, y_tr, y_te,
                                          n_components=n_comp)
        print(f"  {n_comp:>7}  {per['CVL_pls']:>16.4f}  "
              f"{glob['CVL_global_pls']:>12.4f}  {glob['r2']:>13.4f}")

print("\nDone.")
