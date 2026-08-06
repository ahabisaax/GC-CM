"""
Kernel CVL on CUB: RBF random Fourier features on residuals → predict class.
Computes kernel RTV = clipped_kernel_R2 * resid_var and compares to Ridge RTV.
"""
import sys, os
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import r2_score
from sklearn.preprocessing import label_binarize

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)

EMBED_CACHE = master + "results/cub_embeddings_clean_split.joblib"
RIDGE_ALPHA = 1.0
D_RFF       = 512
SEED        = 42

cache = joblib.load(EMBED_CACHE)

for model_name in cache:
    c_mix_tr, c_true_tr, y_tr, c_mix_te, c_true_te, y_te = cache[model_name]
    N_tr, K, m = c_mix_tr.shape
    C = int(y_te.max()) + 1
    classes  = np.arange(C)
    Y_tr_oh  = label_binarize(y_tr, classes=classes).astype(np.float32)
    Y_te_oh  = label_binarize(y_te, classes=classes).astype(np.float32)

    # Step 1: Ridge removes concept-explained variance → residuals
    residuals_tr = np.zeros_like(c_mix_tr)
    residuals_te = np.zeros_like(c_mix_te)
    for k in range(K):
        reg = Ridge(alpha=RIDGE_ALPHA)
        reg.fit(c_true_tr[:, k:k+1], c_mix_tr[:, k, :])
        residuals_tr[:, k, :] = c_mix_tr[:, k, :] - reg.predict(c_true_tr[:, k:k+1])
        residuals_te[:, k, :] = c_mix_te[:, k, :] - reg.predict(c_true_te[:, k:k+1])

    resid_var_per_k = np.array([np.var(residuals_te[:, k, :]) for k in range(K)])
    resid_var_mean  = float(resid_var_per_k.mean())

    # Ridge CVL (current method) for comparison
    ridge_cvl = []
    for k in range(K):
        reg2 = Ridge(alpha=RIDGE_ALPHA)
        reg2.fit(Y_tr_oh, residuals_tr[:, k, :].astype(np.float32))
        r2k = r2_score(residuals_te[:, k, :], reg2.predict(Y_te_oh),
                       multioutput="variance_weighted")
        ridge_cvl.append(r2k)
    ridge_cvl   = np.array(ridge_cvl)
    ridge_rtv   = float(np.mean(np.maximum(0, ridge_cvl)) * resid_var_mean)

    # Median heuristic for gamma — use concept 0's residuals only
    sub_k0 = residuals_tr[:500, 0, :]          # (500, m)
    sq_dists = np.sum((sub_k0[:, None, :] - sub_k0[None, :, :]) ** 2, axis=-1)
    upper = sq_dists[np.triu_indices(500, k=1)]
    gamma  = float(1.0 / (2.0 * np.median(upper) + 1e-8))

    print(f"\n{'='*60}")
    print(f"Model: {model_name}  K={K}  m={m}  gamma={gamma:.5f}")
    print(f"  resid_var_mean = {resid_var_mean:.5f}")
    print(f"  Ridge  CVL clipped = {np.mean(np.maximum(0, ridge_cvl)):.5f}  RTV = {ridge_rtv:.5f}")
    print(f"{'='*60}")

    # Kernel CVL: RBFSampler on residual_k → Ridge → predict Y_onehot
    kernel_cvl = []
    for k in range(K):
        rff  = RBFSampler(gamma=gamma, n_components=D_RFF, random_state=SEED)
        Z_tr = rff.fit_transform(residuals_tr[:, k, :])
        Z_te = rff.transform(residuals_te[:, k, :])
        reg3 = Ridge(alpha=RIDGE_ALPHA)
        reg3.fit(Z_tr, Y_tr_oh)
        r2k  = r2_score(Y_te_oh, reg3.predict(Z_te), multioutput="variance_weighted")
        kernel_cvl.append(r2k)
        if (k + 1) % 20 == 0:
            print(f"  concept {k+1}/{K} done", flush=True)

    kernel_cvl  = np.array(kernel_cvl)
    kernel_clipped = float(np.mean(np.maximum(0, kernel_cvl)))
    kernel_rtv  = kernel_clipped * resid_var_mean

    print(f"\n  Kernel CVL (D={D_RFF} RFF, direction: residual → class):")
    print(f"    raw mean    = {np.mean(kernel_cvl):.5f}")
    print(f"    clipped     = {kernel_clipped:.5f}  neg_frac={np.mean(kernel_cvl < 0):.3f}")
    print(f"    kernel RTV  = {kernel_rtv:.5f}")
    print(f"\n  Comparison:")
    print(f"    Ridge  RTV  = {ridge_rtv:.5f}")
    print(f"    Kernel RTV  = {kernel_rtv:.5f}")
    print(f"    ratio kernel/ridge RTV = {kernel_rtv / (ridge_rtv + 1e-8):.4f}")


print("\nDone.")
