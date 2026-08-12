"""
Sanity check: RTL and RCL should be invariant to appending Gaussian noise
dimensions that are independent of both concepts and task labels.

Tests the current implementation against the invariant (sum) formulation.

Run from project root:
    python experiments/evaluate_models/test_rtl_noise_invariance.py
"""
import os, sys
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import label_binarize

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)

RIDGE_ALPHA = 1.0
NOISE_DIMS  = [0, 8, 16, 32, 64]   # extra noise dims to append
RNG_SEED    = 42


# ---------------------------------------------------------------------------
# Core metric: returns both the current (avg) and invariant (sum) RTL/RCL
# ---------------------------------------------------------------------------

def compute_rtl_rcl(c_mix_tr, c_true_tr, y_tr, c_mix_te, c_true_te, y_te):
    """
    Returns dict with:
        rtl_avg, rcl_avg  — current formulation (average over dims) → NOT invariant
        rtl_sum, rcl_sum  — sum over dims → invariant to appending noise dims
        m                 — embedding dim used
    """
    N_tr, K, m = c_mix_tr.shape
    classes  = np.arange(int(y_te.max()) + 1)
    Y_tr_oh  = label_binarize(y_tr, classes=classes).astype(np.float32)
    Y_te_oh  = label_binarize(y_te, classes=classes).astype(np.float32)

    rtl_per_k_avg = []
    rcl_per_k_avg = []
    rtl_per_k_sum = []
    rcl_per_k_sum = []

    for k in range(K):
        tr_k = c_mix_tr[:, k, :]
        te_k = c_mix_te[:, k, :]

        # Per-dimension normalise (so appending noise dims doesn't rescale signal dims)
        mu    = tr_k.mean(axis=0, keepdims=True)
        sigma = tr_k.std(axis=0, keepdims=True) + 1e-8
        tr_n  = (tr_k - mu) / sigma
        te_n  = (te_k - mu) / sigma

        ck_tr = c_true_tr[:, k]
        ck_te = c_true_te[:, k]

        # Step 1: Ridge(c_k → emb_k) → residual
        reg1     = Ridge(alpha=RIDGE_ALPHA).fit(ck_tr[:, None], tr_n)
        r_tr     = tr_n - reg1.predict(ck_tr[:, None])
        r_te     = te_n - reg1.predict(ck_te[:, None])

        # Per-dimension variance of residual on test
        dim_var  = r_te.var(axis=0)          # [m]
        resid_var_avg = float(dim_var.mean()) # current: average
        resid_var_sum = float(dim_var.sum())  # invariant: sum

        # Step 2a CVL: Ridge(Y → r)
        reg2     = Ridge(alpha=RIDGE_ALPHA).fit(Y_tr_oh, r_tr)
        r_pred   = reg2.predict(Y_te_oh)

        # Per-dimension R²
        ss_res   = ((r_te - r_pred) ** 2).sum(axis=0)  # [m]
        ss_tot   = ((r_te - r_te.mean(axis=0)) ** 2).sum(axis=0)
        r2_per   = np.where(ss_tot > 1e-12, 1 - ss_res / ss_tot, 0.0)  # [m]

        # Current: variance-weighted average R²
        cvl_r2_avg = float(np.average(r2_per, weights=dim_var))
        rtl_avg_k  = max(0.0, cvl_r2_avg) * resid_var_avg

        # Invariant: sum of (var_i × R²_i)
        rtl_sum_k  = float(np.maximum(0, r2_per * dim_var).sum())

        # Step 2b ICVL: Ridge(c_j → r) for j != k, then average
        icvl_avg_list, icvl_sum_list = [], []
        for j in range(K):
            if j == k: continue
            cj_tr = c_true_tr[:, j]; cj_te = c_true_te[:, j]
            reg3   = Ridge(alpha=RIDGE_ALPHA).fit(cj_tr[:, None], r_tr)
            rj_pred = reg3.predict(cj_te[:, None])
            ss_res_j = ((r_te - rj_pred) ** 2).sum(axis=0)
            r2_j     = np.where(ss_tot > 1e-12, 1 - ss_res_j / ss_tot, 0.0)
            icvl_avg_list.append(float(np.average(r2_j, weights=dim_var)))
            icvl_sum_list.append(float((r2_j * dim_var).sum()))

        icvl_r2_avg = float(np.mean(icvl_avg_list)) if icvl_avg_list else 0.0
        icvl_r2_sum = float(np.mean(icvl_sum_list)) if icvl_sum_list else 0.0

        rcl_avg_k = max(0.0, icvl_r2_avg) * resid_var_avg
        rcl_sum_k = float(np.maximum(0, icvl_r2_sum))

        rtl_per_k_avg.append(rtl_avg_k)
        rcl_per_k_avg.append(rcl_avg_k)
        rtl_per_k_sum.append(rtl_sum_k)
        rcl_per_k_sum.append(rcl_sum_k)

    return dict(
        rtl_avg = float(np.mean(rtl_per_k_avg)),
        rcl_avg = float(np.mean(rcl_per_k_avg)),
        rtl_sum = float(np.mean(rtl_per_k_sum)),
        rcl_sum = float(np.mean(rcl_per_k_sum)),
        m       = m,
    )


# ---------------------------------------------------------------------------
# Run test
# ---------------------------------------------------------------------------

CACHE_PATH = master + "results/embeddings_tabulartoy_all.joblib"

if not os.path.exists(CACHE_PATH):
    print(f"Cache not found: {CACHE_PATH}")
    sys.exit(1)

cache = joblib.load(CACHE_PATH)

# Pick one entry: CEM lam_c0.1 fold_1
target_key = next(k for k in cache if "CEM_" in k and "lam_c0.1" in k and "fold_1" in k
                  and "GC" not in k)
data = cache[target_key]
c_mix_tr, c_true_tr, y_tr, c_mix_te, c_true_te, y_te = data
print(f"Entry: {target_key}")
print(f"Embedding shape: {c_mix_tr.shape}  (N, K, m)\n")

rng = np.random.default_rng(RNG_SEED)

print(f"  {'noise_dims':>10}  {'total_m':>7}  {'RTL(avg)':>10}  {'RTL(sum)':>10}  {'RCL(avg)':>10}  {'RCL(sum)':>10}")
print(f"  {'-'*62}")

base = None
for n_noise in NOISE_DIMS:
    if n_noise == 0:
        c_tr = c_mix_tr
        c_te = c_mix_te
    else:
        N_tr, K, m = c_mix_tr.shape
        N_te = c_mix_te.shape[0]
        noise_tr = rng.standard_normal((N_tr, K, n_noise))
        noise_te = rng.standard_normal((N_te, K, n_noise))
        c_tr = np.concatenate([c_mix_tr, noise_tr], axis=2)
        c_te = np.concatenate([c_mix_te, noise_te], axis=2)

    res = compute_rtl_rcl(c_tr, c_true_tr, y_tr, c_te, c_true_te, y_te)

    if base is None:
        base = res

    print(f"  {n_noise:>10}  {res['m']:>7}  {res['rtl_avg']:>10.4f}  {res['rtl_sum']:>10.4f}  {res['rcl_avg']:>10.4f}  {res['rcl_sum']:>10.4f}")

print(f"\n  RTL(avg) with {NOISE_DIMS[-1]} noise dims: {100*(base['rtl_avg'] - compute_rtl_rcl(c_tr, c_true_tr, y_tr, c_te, c_true_te, y_te)['rtl_avg']) / base['rtl_avg']:.1f}% lower than baseline")
print(f"  RTL(sum) is {'invariant' if abs(base['rtl_sum'] - compute_rtl_rcl(c_tr, c_true_tr, y_tr, c_te, c_true_te, y_te)['rtl_sum']) < 0.01 * base['rtl_sum'] else 'NOT invariant'} to noise dimensions")
