"""
CVL/ICVL variants for CUB embeddings: Ridge, probe, PCA, PLS, LDA, partial R².

Run with defaults (clean split, CEM vs GC-CEM emb32):
    python experiments/evaluate_models/cvl_variants_cub.py

Run on a custom embedding cache (e.g. GC-CEM emb16):
    python experiments/evaluate_models/cvl_variants_cub.py \
        --embed_cache results/cub_embeddings_gcem_emb16.joblib \
        --save_path   results/results_cvl_gcem_emb16.dict
"""
import argparse, os, sys
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import r2_score
from sklearn.preprocessing import label_binarize
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--embed_cache", default=master + "results/cub_embeddings_clean_split.joblib")
_ap.add_argument("--save_path",   default=master + "results/results_cvl_variants_cub_clean.dict")
_args, _ = _ap.parse_known_args()

EMBED_CACHE = _args.embed_cache if os.path.isabs(_args.embed_cache) \
              else os.path.join(master, _args.embed_cache)
SAVE_PATH   = _args.save_path if os.path.isabs(_args.save_path) \
              else os.path.join(master, _args.save_path)

PCA_DIMS    = [5, 10, 20, 50]
PLS_DIMS    = [2, 5, 10, 20]
LDA_DIMS    = [2, 5, 10, 20]
RIDGE_ALPHA = 1.0
PROBE_C     = 1.0
PROBE_ITER  = 1000

# ---------------------------------------------------------------------------
cache = joblib.load(EMBED_CACHE)

results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

for model_name in cache.keys():
    c_mix_tr, c_true_tr, y_tr, c_mix_te, c_true_te, y_te = cache[model_name]
    N_tr, K, m = c_mix_tr.shape
    N_te       = c_mix_te.shape[0]
    C          = int(y_te.max()) + 1

    print(f"\n{'='*60}")
    print(f"Model: {model_name}  K={K}  m={m}  C={C}  N_tr={N_tr}  N_te={N_te}")
    print(f"{'='*60}")

    key = model_name
    if key not in results:
        results[key] = {}

    # ------------------------------------------------------------------
    # Step 1 (shared): for each concept k, ridge out c_true[:, k]
    # ------------------------------------------------------------------
    residuals_tr = np.zeros_like(c_mix_tr)   # [N_tr, K, m]
    residuals_te = np.zeros_like(c_mix_te)
    step1_r2s    = []

    for k in range(K):
        reg = Ridge(alpha=RIDGE_ALPHA)
        reg.fit(c_true_tr[:, k:k+1], c_mix_tr[:, k, :])
        pred_tr = reg.predict(c_true_tr[:, k:k+1])
        pred_te = reg.predict(c_true_te[:, k:k+1])
        residuals_tr[:, k, :] = c_mix_tr[:, k, :] - pred_tr
        residuals_te[:, k, :] = c_mix_te[:, k, :] - pred_te
        step1_r2s.append(r2_score(c_mix_te[:, k, :], pred_te,
                                  multioutput="variance_weighted"))

    print(f"  Step-1 R²: mean={np.mean(step1_r2s):.4f}  std={np.std(step1_r2s):.4f}")

    # ------------------------------------------------------------------
    # Original CVL (Ridge vs one-hot y)  — for comparison
    # ------------------------------------------------------------------
    if "cvl_original" not in results[key]:
        classes = np.arange(C)
        Y_tr_oh = label_binarize(y_tr, classes=classes).astype(float)
        Y_te_oh = label_binarize(y_te, classes=classes).astype(float)
        cvl_orig = []
        for k in range(K):
            reg2 = Ridge(alpha=RIDGE_ALPHA)
            reg2.fit(Y_tr_oh, residuals_tr[:, k, :])
            cvl_orig.append(r2_score(residuals_te[:, k, :],
                                     reg2.predict(Y_te_oh),
                                     multioutput="variance_weighted"))
        results[key]["cvl_original"] = {
            "per_concept": cvl_orig,
            "mean":        float(np.mean(cvl_orig)),
            "clipped_mean": float(np.mean(np.maximum(0, cvl_orig))),
            "neg_frac":    float(np.mean(np.array(cvl_orig) < 0)),
        }
        joblib.dump(results, SAVE_PATH)

    r = results[key]["cvl_original"]
    print(f"\n  CVL (original Ridge/one-hot):")
    print(f"    raw mean={r['mean']:.4f}  clipped={r['clipped_mean']:.4f}  neg_frac={r['neg_frac']:.2f}")

    # ------------------------------------------------------------------
    # Variant A: CVL-probe (logistic, above-chance accuracy)
    # ------------------------------------------------------------------
    if "cvl_probe" not in results[key]:
        chance = 1.0 / C
        probe_scores = []
        for k in range(K):
            probe = LogisticRegression(
                C=PROBE_C, max_iter=PROBE_ITER,
                multi_class="multinomial", solver="lbfgs", n_jobs=1,
            )
            probe.fit(residuals_tr[:, k, :], y_tr)
            acc = probe.score(residuals_te[:, k, :], y_te)
            probe_scores.append(max(0.0, acc - chance))

        results[key]["cvl_probe"] = {
            "per_concept": probe_scores,
            "mean":        float(np.mean(probe_scores)),
            "chance":      float(chance),
        }
        joblib.dump(results, SAVE_PATH)

    r = results[key]["cvl_probe"]
    print(f"\n  CVL-probe (above-chance acc, chance={r['chance']:.4f}):")
    print(f"    mean={r['mean']:.4f}  "
          f"min={min(r['per_concept']):.4f}  max={max(r['per_concept']):.4f}")

    # ------------------------------------------------------------------
    # Variant B: CVL-PCA (Ridge vs top-d PCA of one-hot y)
    # ------------------------------------------------------------------
    classes   = np.arange(C)
    Y_tr_oh   = label_binarize(y_tr, classes=classes).astype(float)
    Y_te_oh   = label_binarize(y_te, classes=classes).astype(float)

    for d in PCA_DIMS:
        pca_key = f"cvl_pca_{d}"
        if pca_key in results[key]:
            r = results[key][pca_key]
            print(f"\n  CVL-PCA (d={d:>2}):")
            print(f"    raw mean={r['mean']:.4f}  clipped={r['clipped_mean']:.4f}  neg_frac={r['neg_frac']:.2f}")
            continue

        pca = PCA(n_components=d)
        pca.fit(Y_tr_oh)
        Y_tr_pca = pca.transform(Y_tr_oh)
        Y_te_pca = pca.transform(Y_te_oh)
        var_exp  = pca.explained_variance_ratio_.sum()

        cvl_pca = []
        for k in range(K):
            reg2 = Ridge(alpha=RIDGE_ALPHA)
            reg2.fit(Y_tr_pca, residuals_tr[:, k, :])
            cvl_pca.append(r2_score(residuals_te[:, k, :],
                                    reg2.predict(Y_te_pca),
                                    multioutput="variance_weighted"))

        results[key][pca_key] = {
            "per_concept":  cvl_pca,
            "mean":         float(np.mean(cvl_pca)),
            "clipped_mean": float(np.mean(np.maximum(0, cvl_pca))),
            "neg_frac":     float(np.mean(np.array(cvl_pca) < 0)),
            "var_explained": float(var_exp),
        }
        joblib.dump(results, SAVE_PATH)

        r = results[key][pca_key]
        print(f"\n  CVL-PCA (d={d:>2}, var_exp={var_exp:.3f}):")
        print(f"    raw mean={r['mean']:.4f}  clipped={r['clipped_mean']:.4f}  neg_frac={r['neg_frac']:.2f}")

    # ------------------------------------------------------------------
    # Variant C: CVL-PLS (PLSRegression, finds covariant directions in X and Y)
    # ------------------------------------------------------------------
    classes   = np.arange(C)
    Y_tr_oh   = label_binarize(y_tr, classes=classes).astype(float)
    Y_te_oh   = label_binarize(y_te, classes=classes).astype(float)

    for d in PLS_DIMS:
        pls_key = f"cvl_pls_{d}"
        if pls_key in results[key]:
            r = results[key][pls_key]
            print(f"\n  CVL-PLS (d={d:>2}):")
            print(f"    raw mean={r['mean']:.4f}  clipped={r['clipped_mean']:.4f}  neg_frac={r['neg_frac']:.2f}")
            continue

        cvl_pls = []
        for k in range(K):
            pls = PLSRegression(n_components=d, max_iter=500)
            pls.fit(residuals_tr[:, k, :], Y_tr_oh)
            Y_te_pred = pls.predict(residuals_te[:, k, :])
            cvl_pls.append(r2_score(Y_te_oh, Y_te_pred, multioutput="variance_weighted"))

        results[key][pls_key] = {
            "per_concept":  cvl_pls,
            "mean":         float(np.mean(cvl_pls)),
            "clipped_mean": float(np.mean(np.maximum(0, cvl_pls))),
            "neg_frac":     float(np.mean(np.array(cvl_pls) < 0)),
        }
        joblib.dump(results, SAVE_PATH)

        r = results[key][pls_key]
        print(f"\n  CVL-PLS (d={d:>2}):")
        print(f"    raw mean={r['mean']:.4f}  clipped={r['clipped_mean']:.4f}  neg_frac={r['neg_frac']:.2f}")

    # ------------------------------------------------------------------
    # Variant D: CVL-LDA (project residual through LDA, then above-chance acc)
    # ------------------------------------------------------------------
    # LDA finds the directions in the residual that maximally separate classes.
    # We report above-chance accuracy of the LDA classifier as the metric.
    # n_components capped at min(C-1, m) = min(199, 32) = 32 for CUB.
    max_lda = min(C - 1, m)

    for d in LDA_DIMS:
        d_ = min(d, max_lda)
        lda_key = f"cvl_lda_{d}"
        if lda_key in results[key]:
            r = results[key][lda_key]
            print(f"\n  CVL-LDA (d={d:>2}, actual={d_}):")
            print(f"    above-chance mean={r['mean']:.4f}  min={r['min']:.4f}  max={r['max']:.4f}")
            continue

        chance = 1.0 / C
        lda_scores = []
        for k in range(K):
            lda = LinearDiscriminantAnalysis(n_components=d_, solver="svd",
                                             store_covariance=False)
            lda.fit(residuals_tr[:, k, :], y_tr)
            acc = lda.score(residuals_te[:, k, :], y_te)
            lda_scores.append(max(0.0, acc - chance))

        results[key][lda_key] = {
            "per_concept": lda_scores,
            "mean":        float(np.mean(lda_scores)),
            "min":         float(np.min(lda_scores)),
            "max":         float(np.max(lda_scores)),
            "chance":      float(chance),
            "n_components": d_,
        }
        joblib.dump(results, SAVE_PATH)

        r = results[key][lda_key]
        print(f"\n  CVL-LDA (d={d:>2}, actual={d_}):")
        print(f"    above-chance mean={r['mean']:.4f}  min={r['min']:.4f}  max={r['max']:.4f}")

# ---------------------------------------------------------------------------
# Summary table — adapts column headers to whatever models are in cache
# ---------------------------------------------------------------------------
model_names = list(cache.keys())
col_w = 12
header_cols = "".join(f"  {m:>{col_w}}" for m in model_names)
pos_cols    = "".join(f"  {''+m+'>0?':>{col_w}}" for m in model_names)
sep_w = 27 + (col_w + 2) * len(model_names)

print(f"\n{'='*sep_w}")
print(f"{'Metric':<25}{header_cols}{pos_cols}")
print(f"{'-'*sep_w}")

metric_labels = (
    [("CVL (original)", "cvl_original")]
    + [("CVL-probe", "cvl_probe")]
    + [(f"CVL-PCA d={d}", f"cvl_pca_{d}") for d in PCA_DIMS]
    + [(f"CVL-PLS d={d}", f"cvl_pls_{d}") for d in PLS_DIMS]
    + [(f"CVL-LDA d={d}", f"cvl_lda_{d}") for d in LDA_DIMS]
)

for label, k_ in metric_labels:
    is_acc_metric = k_ in ("cvl_probe",) or k_.startswith("cvl_lda")
    val_cells = []
    pos_cells = []
    for model in model_names:
        if k_ not in results.get(model, {}):
            val_cells.append("N/A")
            pos_cells.append("N/A")
            continue
        r = results[model][k_]
        val = r["mean"] if is_acc_metric else r["clipped_mean"]
        val_cells.append(f"{val:.4f}")
        if is_acc_metric:
            pos_cells.append("yes" if r["mean"] > 0 else "no")
        else:
            pos_cells.append("yes" if r.get("neg_frac", 1) < 0.5 else "no")
    vals_str = "".join(f"  {v:>{col_w}}" for v in val_cells)
    pos_str  = "".join(f"  {p:>{col_w}}" for p in pos_cells)
    print(f"  {label:<23}{vals_str}{pos_str}")

print(f"{'='*sep_w}")
print(f"\nSaved → {SAVE_PATH}")
