"""
Synthetic sanity checks for RTL and RCL (sum formulation).

Three properties tested:
  1. Zero leakage  — RTL ≈ 0, RCL ≈ 0 when embeddings carry no extra info
  2. Monotonicity  — RTL/RCL increase as injected leakage coefficient grows
  3. Noise dims    — RTL/RCL(sum) invariant to appending pure Gaussian noise dims

Data: fully synthetic Gaussian. No real model results needed.

Run from project root:
    python experiments/evaluate_models/test_rtl_sanity.py
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import label_binarize

# ── Config ────────────────────────────────────────────────────────────────────
RIDGE_ALPHA  = 1.0
RNG_SEED     = 42
N_TRAIN      = 3000
N_TEST       = 1000
K            = 4      # concepts
M            = 16     # embedding dims per concept
C            = 4      # task classes (independent of concepts)
NOISE_SIGMA  = 1.0    # std of irreducible noise in embedding


# ── Synthetic data factory ────────────────────────────────────────────────────

def make_data(leakage_rtl: float = 0.0,
              leakage_rcl: float = 0.0,
              n_noise_dims: int = 0,
              rng=None):
    """
    Returns (c_mix_tr, c_true_tr, y_tr, c_mix_te, c_true_te, y_te).

    c_mix has shape (N, K, M + n_noise_dims).

    Embedding for concept k:
        emb_k = c_k * w_k                        (concept signal)
              + leakage_rtl * (y_onehot @ V_k)   (task leakage)
              + leakage_rcl * c_{k+1} * u_k      (inter-concept leakage)
              + noise

    y is drawn uniformly from {0..C-1} and is INDEPENDENT of c_true,
    so Ridge(c_k -> emb_k) cannot absorb any task leakage via c_k↔y correlation.
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)

    N = N_TRAIN + N_TEST

    # Independent binary concepts
    c_true = rng.integers(0, 2, size=(N, K)).astype(np.float32)

    # Task: uniform random, independent of concepts
    y = rng.integers(0, C, size=N).astype(np.int32)
    y_oh = label_binarize(y, classes=np.arange(C)).astype(np.float32)  # (N, C)

    # Fixed random projection matrices (same across all calls via seed)
    W = rng.standard_normal((K, M)).astype(np.float32)      # concept signal
    V = rng.standard_normal((K, C, M)).astype(np.float32)   # task leak direction per k
    U = rng.standard_normal((K, M)).astype(np.float32)      # inter-concept leak direction per k

    total_m = M + n_noise_dims
    c_mix = np.zeros((N, K, total_m), dtype=np.float32)

    for k in range(K):
        j = (k + 1) % K
        signal      = c_true[:, k:k+1] * W[k:k+1, :]       # (N, M)
        task_leak   = leakage_rtl * (y_oh @ V[k])           # (N, M)
        icpt_leak   = leakage_rcl * c_true[:, j:j+1] * U[k:k+1, :]  # (N, M)
        noise       = rng.standard_normal((N, M)).astype(np.float32) * NOISE_SIGMA

        c_mix[:, k, :M] = signal + task_leak + icpt_leak + noise

        if n_noise_dims > 0:
            c_mix[:, k, M:] = rng.standard_normal((N, n_noise_dims)).astype(np.float32)

    tr = slice(0, N_TRAIN)
    te = slice(N_TRAIN, N)
    return (
        c_mix[tr], c_true[tr], y[tr],
        c_mix[te], c_true[te], y[te],
    )


# ── Metric (sum formulation) ──────────────────────────────────────────────────

def compute_rtl_rcl(c_mix_tr, c_true_tr, y_tr, c_mix_te, c_true_te, y_te):
    """
    Returns (rtl_sum, rcl_sum, rtl_norm, rcl_norm), each a mean over concepts.

    rtl_sum / rcl_sum  — noise-invariant absolute measure (units: variance)
        RTL_k = Σᵢ max(0, R²ᵢ(Y→r_k) × varᵢ(r_k))

    rtl_norm / rcl_norm — normalised to [0,1]: fraction of residual variance
        RTL_norm_k = RTL_sum_k / Σᵢ varᵢ(r_k)

    Pipeline per concept k:
      1. Per-dim normalise embedding to unit variance (train stats).
      2. Ridge(c_k → emb_k_norm), take residual r_k.
      3. Per-dim R² of Ridge(Y → r_k) → RTL.
      4. Per-dim R² of Ridge(c_j → r_k) for j≠k, average over j → RCL.
    """
    N_tr, K_, m = c_mix_tr.shape
    classes = np.arange(int(y_te.max()) + 1)
    Y_tr_oh = label_binarize(y_tr, classes=classes).astype(np.float32)
    Y_te_oh = label_binarize(y_te, classes=classes).astype(np.float32)

    rtl_sum_k, rcl_sum_k = [], []
    rtl_norm_k, rcl_norm_k = [], []

    for k in range(K_):
        tr_k = c_mix_tr[:, k, :]
        te_k = c_mix_te[:, k, :]

        # Per-dim normalise (unit variance per dim)
        mu    = tr_k.mean(axis=0, keepdims=True)
        sigma = tr_k.std(axis=0, keepdims=True) + 1e-8
        tr_n  = (tr_k - mu) / sigma
        te_n  = (te_k - mu) / sigma

        # Step 1: subtract concept-predictable part
        reg1 = Ridge(alpha=RIDGE_ALPHA).fit(c_true_tr[:, k:k+1], tr_n)
        r_tr = tr_n - reg1.predict(c_true_tr[:, k:k+1])
        r_te = te_n - reg1.predict(c_true_te[:, k:k+1])

        dim_var      = r_te.var(axis=0)              # [m]
        total_resid  = float(dim_var.sum()) + 1e-12  # denominator for normalisation

        ss_tot = ((r_te - r_te.mean(axis=0)) ** 2).sum(axis=0)  # [m]

        # Step 2a — RTL: Ridge(Y → r)
        reg2   = Ridge(alpha=RIDGE_ALPHA).fit(Y_tr_oh, r_tr)
        r_pred = reg2.predict(Y_te_oh)
        ss_res = ((r_te - r_pred) ** 2).sum(axis=0)
        r2_y   = np.where(ss_tot > 1e-12, 1 - ss_res / ss_tot, 0.0)

        rtl_k_sum  = float(np.maximum(0.0, r2_y * dim_var).sum())
        rtl_k_norm = rtl_k_sum / total_resid

        # Step 2b — RCL: Ridge(c_j → r) for j≠k, average over j
        rcl_j_sum, rcl_j_norm = [], []
        for j in range(K_):
            if j == k:
                continue
            reg3     = Ridge(alpha=RIDGE_ALPHA).fit(c_true_tr[:, j:j+1], r_tr)
            rj_pred  = reg3.predict(c_true_te[:, j:j+1])
            ss_res_j = ((r_te - rj_pred) ** 2).sum(axis=0)
            r2_j     = np.where(ss_tot > 1e-12, 1 - ss_res_j / ss_tot, 0.0)
            v        = float(np.maximum(0.0, r2_j * dim_var).sum())
            rcl_j_sum.append(v)
            rcl_j_norm.append(v / total_resid)

        rtl_sum_k.append(rtl_k_sum)
        rtl_norm_k.append(rtl_k_norm)
        rcl_sum_k.append(float(np.mean(rcl_j_sum)) if rcl_j_sum else 0.0)
        rcl_norm_k.append(float(np.mean(rcl_j_norm)) if rcl_j_norm else 0.0)

    return (
        float(np.mean(rtl_sum_k)),
        float(np.mean(rcl_sum_k)),
        float(np.mean(rtl_norm_k)),
        float(np.mean(rcl_norm_k)),
    )


# ── Test 1: Zero leakage ──────────────────────────────────────────────────────

def test_zero_leakage():
    print("=" * 60)
    print("TEST 1: Zero leakage → RTL ≈ 0, RCL ≈ 0")
    print("=" * 60)
    data = make_data(leakage_rtl=0.0, leakage_rcl=0.0)
    rtl_s, rcl_s, rtl_n, rcl_n = compute_rtl_rcl(*data)
    tol_sum, tol_norm = 0.02, 0.005
    print(f"  {'':20s}  {'sum':>10}  {'norm [0,1]':>10}")
    print(f"  {'RTL':20s}  {rtl_s:>10.5f}  {rtl_n:>10.5f}  (tol {tol_sum}/{tol_norm})")
    print(f"  {'RCL':20s}  {rcl_s:>10.5f}  {rcl_n:>10.5f}")
    ok = rtl_s < tol_sum and rcl_s < tol_sum and rtl_n < tol_norm and rcl_n < tol_norm
    print(f"  {'PASS' if ok else 'FAIL'}\n")
    return ok


# ── Test 2: Monotonicity with injected leakage ────────────────────────────────

def test_monotonicity():
    print("=" * 60)
    print("TEST 2: Monotonicity — scores increase with injected leakage")
    print("=" * 60)

    coefs = [0.0, 0.5, 1.0, 2.0, 4.0]
    hdr   = f"  {'coef':>6}  {'RTL(sum)':>10}  {'RTL(norm)':>10}  {'RCL(sum)':>10}  {'RCL(norm)':>10}"

    print("\n  --- RTL (varying task leakage, RCL=0) ---")
    print(hdr)
    rtl_s_vals, rtl_n_vals = [], []
    for lam in coefs:
        data = make_data(leakage_rtl=lam, leakage_rcl=0.0)
        rs, _, rn, _ = compute_rtl_rcl(*data)
        rtl_s_vals.append(rs); rtl_n_vals.append(rn)
        print(f"  {lam:>6.1f}  {rs:>10.5f}  {rn:>10.5f}")
    rtl_s_mono = all(rtl_s_vals[i] <= rtl_s_vals[i+1] for i in range(len(rtl_s_vals)-1))
    rtl_n_mono = all(rtl_n_vals[i] <= rtl_n_vals[i+1] for i in range(len(rtl_n_vals)-1))
    print(f"  RTL(sum) monotone: {'PASS' if rtl_s_mono else 'FAIL'}")
    print(f"  RTL(norm) monotone: {'PASS' if rtl_n_mono else 'FAIL'}")

    print(f"\n  --- RCL (varying inter-concept leakage, RTL=0) ---")
    print(hdr)
    rcl_s_vals, rcl_n_vals = [], []
    for lam in coefs:
        data = make_data(leakage_rtl=0.0, leakage_rcl=lam)
        _, rs, _, rn = compute_rtl_rcl(*data)
        rcl_s_vals.append(rs); rcl_n_vals.append(rn)
        print(f"  {lam:>6.1f}  {rs:>10.5f}  {rn:>10.5f}")
    rcl_s_mono = all(rcl_s_vals[i] <= rcl_s_vals[i+1] for i in range(len(rcl_s_vals)-1))
    rcl_n_mono = all(rcl_n_vals[i] <= rcl_n_vals[i+1] for i in range(len(rcl_n_vals)-1))
    print(f"  RCL(sum) monotone: {'PASS' if rcl_s_mono else 'FAIL'}")
    print(f"  RCL(norm) monotone: {'PASS' if rcl_n_mono else 'FAIL'}\n")

    return rtl_s_mono and rtl_n_mono and rcl_s_mono and rcl_n_mono


# ── Test 3: Noise dimension invariance ────────────────────────────────────────

def test_noise_invariance():
    print("=" * 60)
    print("TEST 3: Noise dims — RTL/RCL(sum) invariant; norm dilutes")
    print("=" * 60)
    leakage_rtl = 2.0
    leakage_rcl = 2.0
    noise_dims   = [0, 8, 16, 32, 64]

    print(f"  (leakage_rtl={leakage_rtl}, leakage_rcl={leakage_rcl})")
    print(f"  {'noise_dims':>10}  {'total_m':>7}  {'RTL(sum)':>10}  {'RTL(norm)':>10}  {'RCL(sum)':>10}  {'RCL(norm)':>10}")

    base = None
    rtl_s_vals, rcl_s_vals = [], []
    for nd in noise_dims:
        data = make_data(leakage_rtl=leakage_rtl, leakage_rcl=leakage_rcl, n_noise_dims=nd)
        rtl_s, rcl_s, rtl_n, rcl_n = compute_rtl_rcl(*data)
        if base is None:
            base = (rtl_s, rcl_s)
        rtl_s_vals.append(rtl_s); rcl_s_vals.append(rcl_s)
        print(f"  {nd:>10}  {M + nd:>7}  {rtl_s:>10.5f}  {rtl_n:>10.5f}  {rcl_s:>10.5f}  {rcl_n:>10.5f}")

    tol = 0.05
    rtl_inv = all(abs(v - base[0]) / (base[0] + 1e-9) < tol for v in rtl_s_vals)
    rcl_inv = all(abs(v - base[1]) / (base[1] + 1e-9) < tol for v in rcl_s_vals)
    print(f"  RTL(sum) invariant: {'PASS' if rtl_inv else 'FAIL'}")
    print(f"  RCL(sum) invariant: {'PASS' if rcl_inv else 'FAIL'}")
    print(f"  RTL(norm) decreases with noise dims (expected — see note below)")
    print(f"  Note: norm divides by total residual var, which grows with noise dims.\n")
    return rtl_inv and rcl_inv


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    r1 = test_zero_leakage()
    r2 = test_monotonicity()
    r3 = test_noise_invariance()

    print("=" * 60)
    overall = all([r1, r2, r3])
    print(f"OVERALL: {'ALL PASS' if overall else 'SOME FAILED'}")
