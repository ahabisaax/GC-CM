"""
Controlled Gaussian validation of CVL and ICVL metrics.

Setup
-----
K binary concepts c_k ~ Bernoulli(0.5), independent.
Embedding for concept k: ĉ_k ~ N(μ_{c_k, k}, I_m)  [base — no leakage]

CVL experiment  (task leakage):
    ĉ_k += α · φ_y        where φ_y ∈ ℝ^m is a class-specific offset.
    Expected: CVL ≈ 0 at α=0, grows monotonically with α.

ICVL experiment (inter-concept leakage):
    ĉ_1 += β · ψ_{c_0}   injecting concept-0 label into concept-1's embedding.
    Expected: ICVL ≈ 0 at β=0, grows monotonically with β.

Run from project root:
    python experiments/evaluate_models/validate_cvl_icvl.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xai_concept_leakage.metrics.leakage import compute_CVL, compute_ICVL

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
N       = 3000    # total samples
K       = 6       # number of concepts
M       = 16      # embedding dimension per concept
N_TRAIN = 2000
N_TEST  = N - N_TRAIN
N_SEEDS = 20
ALPHAS  = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])

# Fixed prototypes and signal vectors (shared across seeds and α levels)
_proto_rng = np.random.RandomState(0)
MU_POS   = _proto_rng.randn(K, M)        # [K, M] positive prototypes
MU_NEG   = _proto_rng.randn(K, M)        # [K, M] negative prototypes
TASK_SIG = _proto_rng.randn(2, M)        # [2, M] per-class task signal
IC_SIG_0 = _proto_rng.randn(M)           # signal for c_0=0 injected into ĉ_1
IC_SIG_1 = _proto_rng.randn(M)           # signal for c_0=1 injected into ĉ_1

# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------

def make_embeddings(alpha_task: float, alpha_ic: float, seed: int):
    """
    Generate (c_hat_train, c_hat_test, c_train, c_test, y_train, y_test).

    alpha_task : strength of task-label leakage injected into ALL concepts
    alpha_ic   : strength of concept-0 leakage injected into concept 1 only
    """
    rng = np.random.RandomState(seed)

    # Binary concepts, independent
    c = rng.randint(0, 2, (N, K)).astype(float)          # [N, K]

    # Task: majority vote → binary
    y = (c.sum(axis=1) >= K / 2).astype(int)             # [N]

    # Build [N, K, M] embedding array
    c_hat = np.zeros((N, K, M))
    for k in range(K):
        base      = c[:, k:k+1] * MU_POS[k] + (1 - c[:, k:k+1]) * MU_NEG[k]
        task_leak = alpha_task * TASK_SIG[y]              # [N, M]
        ic_leak   = np.zeros((N, M))
        if k == 1:                                         # inject c_0 into ĉ_1
            ic_leak = alpha_ic * (
                c[:, 0:1] * IC_SIG_1 + (1 - c[:, 0:1]) * IC_SIG_0
            )
        noise           = rng.randn(N, M)
        c_hat[:, k, :]  = base + task_leak + ic_leak + noise

    # Train / test split
    idx = rng.permutation(N)
    tr, te = idx[:N_TRAIN], idx[N_TRAIN:]
    return (
        c_hat[tr], c_hat[te],
        c[tr], c[te],
        y[tr], y[te],
    )


# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------
print("=== CVL validation (task leakage) ===")
cvl_means, cvl_stds = [], []
for alpha in ALPHAS:
    vals = []
    for seed in range(N_SEEDS):
        tr_hat, te_hat, c_tr, c_te, y_tr, y_te = make_embeddings(alpha, 0.0, seed)
        res = compute_CVL(tr_hat, te_hat, c_tr, c_te, y_tr, y_te)
        vals.append(res["CVL"])
    m, s = np.mean(vals), np.std(vals)
    cvl_means.append(m); cvl_stds.append(s)
    print(f"  α={alpha:.2f}  CVL={m:.4f} ± {s:.4f}")

print("\n=== ICVL validation (inter-concept leakage) ===")
icvl_means, icvl_stds = [], []
for alpha in ALPHAS:
    vals = []
    for seed in range(N_SEEDS):
        tr_hat, te_hat, c_tr, c_te, y_tr, y_te = make_embeddings(0.0, alpha, seed)
        res = compute_ICVL(tr_hat, te_hat, c_tr, c_te)
        vals.append(res["ICVL"])
    m, s = np.mean(vals), np.std(vals)
    icvl_means.append(m); icvl_stds.append(s)
    print(f"  β={alpha:.2f}  ICVL={m:.4f} ± {s:.4f}")

cvl_means  = np.array(cvl_means);  cvl_stds  = np.array(cvl_stds)
icvl_means = np.array(icvl_means); icvl_stds = np.array(icvl_stds)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
C_CVL  = "#0072B2"   # blue
C_ICVL = "#D55E00"   # vermillion

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

for ax, means, stds, color, xlabel, ylabel, title in [
    (axes[0], cvl_means,  cvl_stds,  C_CVL,
     r"Task leakage strength $\alpha$", "CVL",
     "CVL vs task leakage strength"),
    (axes[1], icvl_means, icvl_stds, C_ICVL,
     r"Inter-concept leakage strength $\beta$", "ICVL",
     "ICVL vs inter-concept leakage strength"),
]:
    ax.plot(ALPHAS, means, color=color, lw=2, marker="o", ms=5, zorder=3)
    ax.fill_between(ALPHAS, means - stds, means + stds,
                    color=color, alpha=0.18, zorder=2)
    ax.axhline(0, color="0.5", lw=0.8, ls="--", zorder=1)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    ax.set_xlim(ALPHAS[0] - 0.1, ALPHAS[-1] + 0.1)
    ax.set_ylim(bottom=-0.02)

    # Annotate zero point
    ax.annotate("No leakage\n(α = 0)" if "Task" in xlabel else "No leakage\n(β = 0)",
                xy=(0, means[0]),
                xytext=(0.4, means[0] + 0.02 * (means[-1] - means[0])),
                fontsize=8, color="0.4",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))

fig.suptitle(
    f"Controlled Gaussian validation  "
    f"(K={K} concepts, m={M} dims, N={N_TRAIN}+{N_TEST}, {N_SEEDS} seeds)",
    fontsize=12, fontweight="bold",
)

master_folder = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
PLOT_DIR = master_folder + "results/plots/cbm/"
os.makedirs(PLOT_DIR, exist_ok=True)
out = PLOT_DIR + "validate_cvl_icvl.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.close(fig)
