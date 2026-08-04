"""
Experiment: how CVL, ICVL, CTL, and ICL vary as the step-1 explained variance
changes.

Data model (same as cvl_robustness_experiment.py):

    e_k[n, j] = SIGNAL_C  * c_k[n]              (concept signal  — varied)
              + SIGNAL_IC * c_{(k+1)%K}[n]       (inter-concept leakage  — fixed)
              + gamma_k[j, y[n]]                  (task leakage — fixed; gamma ~ N(0,1))
              + SIGMA * noise_j                   (noise — fixed)

SIGNAL_C is swept from low to high, which controls how much of the embedding
variance is explained by the concept in step 1 of CVL/ICVL computation.

Step-1 R² (analytical):
    R²_step1 = 0.25·SIGNAL_C² / (0.25·SIGNAL_C² + 0.25·SIGNAL_IC² + 1 + SIGMA²)

True CVL and ICVL are INDEPENDENT of SIGNAL_C (step 1 is perfect analytically),
but depend on C and K (the C→∞ formula differs from finite C):
    _var_gamma = (C-1)/C            # expected sample variance of C N(0,1) values
    _var_resid = SIGNAL_IC²·0.25 + _var_gamma + SIGMA²
    TRUE_CVL   = _var_gamma / _var_resid
    TRUE_ICVL  = (SIGNAL_IC²·0.25 / _var_resid) / (K-1)   (one adjacent pair per concept)

For C=2, K=6, SIGNAL_IC=2, SIGMA=1:
    TRUE_CVL = 0.5/2.5 = 0.200,  TRUE_ICVL = (1.0/2.5)/5 = 0.080

KSG CTL / ICL use the training-time differential formula:
    CTL = max(0, MI(e_flat, y) − MI(c_true, y))
    ICL = max(0, mean(MI(e_pairs)) − mean(MI(c_pairs)))

Since y is independent of c in the data generator (task signal is injected
directly into the embedding via gamma_k), MI(c_true, y) ≈ 0.  So CTL ≈ MI(e_flat, y),
which depends only on the task signal — not on SIGNAL_C.  This gives KSG a
flat expected profile, in contrast to CVL/ICVL which degrade as step-1 quality
drops.

Run with:
    conda run -n xai-leakage python experiments/evaluate_models/cvl_step1_quality_experiment.py
"""
import os, sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)

from xai_concept_leakage.metrics.mutual_information import compute_mi_matrix_parallel

SAVE_PATH = master + "results/results_cvl_step1_quality.dict"

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------
K        = 6       # concepts (small so KSG is fast: 6*8=48 dims, 1128 pairs)
C        = 2       # binary task (y independent of c — task injected via gamma)
EMB_SIZE = 8       # embedding dims per concept
N_TRAIN  = 3000
N_TEST   = 1000
ALPHA    = 1.0     # ridge regularisation
SEEDS    = 3
N_KSG_NEIGHBORS = 3
MAX_KSG_SAMPLES = 2000  # subsample cap for KSG speed

SIGNAL_IC = 2.0
SIGMA     = 1.0

# Sweep: SIGNAL_C controls step-1 quality
SIGNAL_C_VALS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def true_step1_r2(sig_c):
    return (0.25 * sig_c**2
            / (0.25 * sig_c**2 + 0.25 * SIGNAL_IC**2 + 1.0 + SIGMA**2))


# Analytical ground truth for ridge-based metrics (independent of SIGNAL_C,
# but depends on C and K — derived from the data model).
#
# After step 1, residual R_k[j] = SIGNAL_IC·c_{k+1} + gamma_k[j,y] + SIGMA·noise
# Var_y(gamma_k[j,y]) = sample variance of C N(0,1) values = (C-1)/C in expectation
#   → for C=2: 0.5; for C→∞: 1.0  (the original formula assumed C→∞)
#
# TRUE_CVL  = E[Var(E[R|y])] / E[Var(R)]
#           = (C-1)/C  /  (SIGNAL_IC²·0.25 + (C-1)/C + SIGMA²)
#
# TRUE_ICVL = Var(SIGNAL_IC·c_{k+1}) / Var(R),  for the one adjacent pair per concept,
#             averaged over all K·(K-1) off-diagonal pairs (K-1 non-adjacent → 0).
#           = (SIGNAL_IC²·0.25 / Var(R)) / (K-1)
_var_gamma = (C - 1) / C
_var_resid = 0.25 * SIGNAL_IC**2 + _var_gamma + SIGMA**2
TRUE_CVL   = _var_gamma / _var_resid
TRUE_ICVL  = (0.25 * SIGNAL_IC**2 / _var_resid) / (K - 1)


# ---------------------------------------------------------------------------
# Data generation  (identical structure to cvl_robustness_experiment.py)
# ---------------------------------------------------------------------------
def generate_data(N, sig_c, rng):
    c = rng.randint(0, 2, (N, K)).astype(float)   # [N, K] i.i.d. Bernoulli(0.5)
    y = rng.randint(0, C, N)                        # [N]   independent of c

    gamma = [rng.randn(EMB_SIZE, C) for _ in range(K)]

    emb = np.zeros((N, K, EMB_SIZE))
    for k in range(K):
        k_next = (k + 1) % K
        emb[:, k, :] = (
            sig_c     * np.outer(c[:, k],      np.ones(EMB_SIZE))
            + SIGNAL_IC * np.outer(c[:, k_next], np.ones(EMB_SIZE))
            + gamma[k][:, y].T
            + SIGMA   * rng.randn(N, EMB_SIZE)
        )
    return emb, c, y


# ---------------------------------------------------------------------------
# Ridge-based CVL and ICVL
# ---------------------------------------------------------------------------
def _residuals_step1(emb_tr, emb_te, c_tr, c_te, k):
    """Residuals after regressing c_k → emb[:, k]."""
    reg = Ridge(alpha=ALPHA)
    reg.fit(c_tr[:, k:k+1], emb_tr[:, k])
    pred_tr = reg.predict(c_tr[:, k:k+1])
    pred_te = reg.predict(c_te[:, k:k+1])
    r2 = r2_score(emb_te[:, k], pred_te, multioutput="variance_weighted")
    return emb_tr[:, k] - pred_tr, emb_te[:, k] - pred_te, float(r2)


def estimate_cvl_icvl(emb_tr, emb_te, c_tr, c_te, y_tr, y_te):
    """
    Returns (cvl, icvl, mean_step1_r2).
    CVL uses one-hot y (correct for C=2 it is the same as scalar).
    """
    Y_tr = y_tr.reshape(-1, 1).astype(float)
    Y_te = y_te.reshape(-1, 1).astype(float)

    cvl_vals, icvl_vals, step1_r2s = [], [], []

    # Pre-compute residuals for all concepts
    residuals_tr, residuals_te = {}, {}
    for k in range(K):
        R_tr, R_te, s1r2 = _residuals_step1(emb_tr, emb_te, c_tr, c_te, k)
        residuals_tr[k] = R_tr
        residuals_te[k] = R_te
        step1_r2s.append(s1r2)

        # CVL: regress y → R_k
        reg = Ridge(alpha=ALPHA)
        reg.fit(Y_tr, R_tr)
        cvl_vals.append(max(0.0, r2_score(R_te, reg.predict(Y_te),
                                           multioutput="variance_weighted")))

    # ICVL: regress c_i → R_k for i ≠ k
    for k in range(K):
        for i in range(K):
            if i == k:
                continue
            reg = Ridge(alpha=ALPHA)
            reg.fit(c_tr[:, i:i+1], residuals_tr[k])
            icvl_vals.append(max(0.0,
                r2_score(residuals_te[k], reg.predict(c_te[:, i:i+1]),
                         multioutput="variance_weighted")))

    return float(np.mean(cvl_vals)), float(np.mean(icvl_vals)), float(np.mean(step1_r2s))


# ---------------------------------------------------------------------------
# KSG CTL / ICL  (matches training-time formula from train/utils.py)
# ---------------------------------------------------------------------------
def estimate_ctl_icl(emb, c, y, n_jobs):
    N = len(y)
    e_flat = emb.reshape(N, -1)    # [N, K * EMB_SIZE]

    if N > MAX_KSG_SAMPLES:
        idx    = np.random.choice(N, MAX_KSG_SAMPLES, replace=False)
        e_flat = e_flat[idx]
        c      = c[idx]
        y      = y[idx]

    # CTL = max(0, MI(e_flat, y) − MI(c_true, y))
    mi_e_y = compute_mi_matrix_parallel(
        e_flat, d=y,
        n_neighbors=N_KSG_NEIGHBORS, normalise=True, flatten=False, n_jobs=n_jobs,
    )
    mi_c_y = compute_mi_matrix_parallel(
        c, d=y,
        n_neighbors=N_KSG_NEIGHBORS, normalise=True, flatten=False, n_jobs=n_jobs,
    )
    ctl = float(np.maximum(0, np.mean(mi_e_y) - np.mean(mi_c_y)))

    # ICL = max(0, mean(MI(e pairs)) − mean(MI(c pairs)))
    mi_e_pairs = compute_mi_matrix_parallel(
        e_flat,
        n_neighbors=N_KSG_NEIGHBORS, normalise=True, flatten=True, n_jobs=n_jobs,
    )
    mi_c_pairs = compute_mi_matrix_parallel(
        c,
        n_neighbors=N_KSG_NEIGHBORS, normalise=True, flatten=True, n_jobs=n_jobs,
    )
    icl = float(np.maximum(0, np.mean(mi_e_pairs) - np.mean(mi_c_pairs)))

    return ctl, icl


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}
n_jobs  = os.cpu_count() or 1

print(f"K={K}  C={C}  EMB_SIZE={EMB_SIZE}  N_TRAIN={N_TRAIN}  seeds={SEEDS}")
print(f"SIGNAL_IC={SIGNAL_IC}  SIGMA={SIGMA}")
print(f"TRUE_CVL={TRUE_CVL:.3f}  TRUE_ICVL={TRUE_ICVL:.3f}")
print(f"Sweep: SIGNAL_C={SIGNAL_C_VALS}")
print(f"Analytical step-1 R²: {[round(true_step1_r2(s), 3) for s in SIGNAL_C_VALS]}")
print(f"n_jobs={n_jobs}\n")

for sig_c in SIGNAL_C_VALS:
    key    = str(sig_c)
    cached = results.get(key, [])
    if len(cached) >= SEEDS:
        cvl  = np.mean([x["cvl"]      for x in cached])
        icvl = np.mean([x["icvl"]     for x in cached])
        ctl  = np.mean([x["ctl"]      for x in cached])
        icl  = np.mean([x["icl"]      for x in cached])
        r2   = np.mean([x["step1_r2"] for x in cached])
        print(f"SIGNAL_C={sig_c:5.2f}  R²={r2:.3f}  "
              f"CVL={cvl:.3f}  ICVL={icvl:.3f}  "
              f"CTL={ctl:.4f}  ICL={icl:.4f}  [cached]")
        continue

    print(f"\nSIGNAL_C={sig_c:.2f}  (expected R²≈{true_step1_r2(sig_c):.3f})", flush=True)
    seed_res = []
    for seed in range(SEEDS):
        rng = np.random.RandomState(seed * 7919 + int(sig_c * 1000))
        emb, c, y = generate_data(N_TRAIN + N_TEST, sig_c, rng)

        emb_tr, c_tr, y_tr = emb[:N_TRAIN], c[:N_TRAIN], y[:N_TRAIN]
        emb_te, c_te, y_te = emb[N_TRAIN:], c[N_TRAIN:], y[N_TRAIN:]

        cvl_v, icvl_v, s1r2 = estimate_cvl_icvl(emb_tr, emb_te, c_tr, c_te, y_tr, y_te)
        ctl_v, icl_v        = estimate_ctl_icl(emb_tr, c_tr, y_tr, n_jobs)

        seed_res.append({
            "cvl":      cvl_v,
            "icvl":     icvl_v,
            "ctl":      ctl_v,
            "icl":      icl_v,
            "step1_r2": s1r2,
        })
        print(f"  seed={seed}  R²={s1r2:.3f}  "
              f"CVL={cvl_v:.3f}  ICVL={icvl_v:.3f}  "
              f"CTL={ctl_v:.4f}  ICL={icl_v:.4f}")

    results[key] = seed_res
    joblib.dump(results, SAVE_PATH)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
print("\nGenerating plot ...")

present = [s for s in SIGNAL_C_VALS if results.get(str(s))]
x_r2    = np.array([np.mean([r["step1_r2"] for r in results[str(s)]]) for s in present])
order   = np.argsort(x_r2)
x_r2    = x_r2[order]


def _agg(metric):
    arr = np.array([[r[metric] for r in results[str(s)]] for s in present])
    arr = arr[order]
    return arr.mean(axis=1), arr.std(axis=1)


cvl_m,  cvl_s  = _agg("cvl")
icvl_m, icvl_s = _agg("icvl")
ctl_m,  ctl_s  = _agg("ctl")
icl_m,  icl_s  = _agg("icl")

BLUE, RED = "#2166ac", "#d6604d"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)


def _panel(ax, ys, colors, labels, linestyles, true_vals, true_labels, ylabel, title):
    for (m, s), col, lab, ls in zip(ys, colors, labels, linestyles):
        ax.plot(x_r2, m, marker="o", color=col, linestyle=ls, linewidth=2, label=lab)
        ax.fill_between(x_r2, m - s, m + s, color=col, alpha=0.18, linewidth=0)
    for tv, tlab, col in zip(true_vals, true_labels, colors):
        ax.axhline(tv, color=col, linestyle=":", linewidth=1.3, label=tlab)
    ax.set_xlabel("Step-1 R²  (c_k explains embedding_k)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)


_panel(
    axes[0],
    ys=[(cvl_m, cvl_s), (icvl_m, icvl_s)],
    colors=[BLUE, RED],
    labels=["CVL (estimated)", "ICVL (estimated)"],
    linestyles=["-", "--"],
    true_vals=[TRUE_CVL, TRUE_ICVL],
    true_labels=[f"true CVL = {TRUE_CVL:.3f}", f"true ICVL = {TRUE_ICVL:.3f}"],
    ylabel="Metric value",
    title="Ridge-based metrics vs step-1 quality",
)

_panel(
    axes[1],
    ys=[(ctl_m, ctl_s), (icl_m, icl_s)],
    colors=[BLUE, RED],
    labels=["CTL (KSG)", "ICL (KSG)"],
    linestyles=["-", "--"],
    true_vals=[],
    true_labels=[],
    ylabel="Metric value",
    title="KSG-based metrics vs step-1 quality",
)
axes[1].text(
    0.97, 0.97,
    "No analytical truth line:\nKSG measures MI in embedding\n"
    "(should be ≈ flat — robust to step-1 quality)",
    transform=axes[1].transAxes, ha="right", va="top",
    fontsize=7.5, color="#555555",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8),
)

plt.suptitle(
    f"Effect of step-1 explained variance on leakage metrics\n"
    f"K={K}, C={C}, emb_size={EMB_SIZE}, "
    f"SIGNAL_IC={SIGNAL_IC}, SIGMA={SIGMA}",
    fontsize=9.5,
)
plt.tight_layout()

for ext in (".pdf", ".png"):
    out = master + f"experiments/evaluate_models/cvl_step1_quality{ext}"
    plt.savefig(out, bbox_inches="tight", dpi=150 if ext == ".png" else None)
    print(f"Saved → {out}")
