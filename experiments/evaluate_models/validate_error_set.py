"""
Controlled Gaussian validation of the error-set analysis.

Setup
-----
K binary concepts c_k ~ Bernoulli(0.5), independent.
Embedding for concept k: ĉ_k = base(c_k) + α·φ_y + noise

where φ_y ∈ ℝ^m is a class-specific task-label offset (leakage signal).

Two sklearn classifiers are fit on the embeddings:
  - Concept predictor: logistic regression  ĉ_k → c_k  (per concept)
  - Task predictor:    logistic regression  [ĉ_1,...,ĉ_K] → y  (joint embedding)

At test time, samples are bucketed by n_wrong = #{k : c_pred_k ≠ c_true_k}.
Task accuracy is reported within each bucket.

Expected result:
  α = 0  →  task accuracy drops steeply with n_wrong (concepts are the only
             path to y — no leaked signal in the embedding).
  high α →  task accuracy stays high even when many concepts are wrong
             (φ_y is baked into the embedding; the task predictor reads it
             directly, bypassing concept correctness).

Run from project root:
    python experiments/evaluate_models/validate_error_set.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
N       = 3000
K       = 6
M       = 16
N_TRAIN = 2000
N_TEST  = N - N_TRAIN
N_SEEDS = 20
ALPHAS  = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

# Fixed prototypes and signal vectors (same seed as validate_cvl_icvl.py)
_proto_rng = np.random.RandomState(0)
MU_POS   = _proto_rng.randn(K, M)
MU_NEG   = _proto_rng.randn(K, M)
TASK_SIG = _proto_rng.randn(2, M)   # [2, M] per-class task signal
_proto_rng.randn(M); _proto_rng.randn(M)  # consume IC_SIG slots for consistency


# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------
def make_embeddings(alpha_task: float, seed: int):
    rng = np.random.RandomState(seed)
    c = rng.randint(0, 2, (N, K)).astype(float)
    y = (c.sum(axis=1) >= K / 2).astype(int)

    c_hat = np.zeros((N, K, M))
    for k in range(K):
        base      = c[:, k:k+1] * MU_POS[k] + (1 - c[:, k:k+1]) * MU_NEG[k]
        task_leak = alpha_task * TASK_SIG[y]
        noise     = rng.randn(N, M)
        c_hat[:, k, :] = base + task_leak + noise

    idx = rng.permutation(N)
    tr, te = idx[:N_TRAIN], idx[N_TRAIN:]
    return c_hat[tr], c_hat[te], c[tr].astype(int), c[te].astype(int), y[tr], y[te]


# ---------------------------------------------------------------------------
# Single-seed error-set analysis
# ---------------------------------------------------------------------------
def run_error_set(alpha_task: float, seed: int):
    tr_hat, te_hat, c_tr, c_te, y_tr, y_te = make_embeddings(alpha_task, seed)

    # Concept predictor: one logistic regression per concept
    c_pred = np.zeros((N_TEST, K), dtype=int)
    for k in range(K):
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(tr_hat[:, k, :], c_tr[:, k])
        c_pred[:, k] = clf.predict(te_hat[:, k, :])

    # Task predictor: trained on full concatenated embedding (mirrors c2y using c_mix)
    task_clf = LogisticRegression(max_iter=500, C=1.0)
    task_clf.fit(tr_hat.reshape(N_TRAIN, -1), y_tr)
    y_pred = task_clf.predict(te_hat.reshape(N_TEST, -1))

    # Bucket by number of wrongly predicted concepts
    n_wrong = (c_pred != c_te).sum(axis=1)
    buckets = {}
    for k in range(K + 1):
        mask = n_wrong == k
        if mask.sum() == 0:
            continue
        buckets[k] = float((y_pred[mask] == y_te[mask]).mean())
    return buckets


# ---------------------------------------------------------------------------
# Sweep over alphas and seeds
# ---------------------------------------------------------------------------
print(f"Error-set analysis  (K={K}, M={M}, N={N_TRAIN}+{N_TEST}, {N_SEEDS} seeds)")
print(f"{'α':>6}  " + "  ".join(f"n_wrong={k:d}" for k in range(K + 1)))

# results[alpha][n_wrong] = list of task_acc values across seeds
all_results = {a: {k: [] for k in range(K + 1)} for a in ALPHAS}

for alpha in ALPHAS:
    for seed in range(N_SEEDS):
        buckets = run_error_set(alpha, seed)
        for k, acc in buckets.items():
            all_results[alpha][k].append(acc)

    means = {k: np.mean(v) if v else np.nan for k, v in all_results[alpha].items()}
    row = f"{alpha:>6.1f}  " + "  ".join(
        f"{means[k]:.3f}" if not np.isnan(means[k]) else "  -- "
        for k in range(K + 1)
    )
    print(row)


# ---------------------------------------------------------------------------
# Plot: one curve per alpha, x = n_wrong, y = task accuracy
# ---------------------------------------------------------------------------
CMAP   = plt.cm.plasma
colors = [CMAP(i / (len(ALPHAS) - 1)) for i in range(len(ALPHAS))]

fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

for alpha, color in zip(ALPHAS, colors):
    xs, ys, yerrs = [], [], []
    for k in range(K + 1):
        vals = all_results[alpha][k]
        if len(vals) < 2:
            continue
        xs.append(k)
        ys.append(np.mean(vals))
        yerrs.append(np.std(vals))
    xs, ys, yerrs = np.array(xs), np.array(ys), np.array(yerrs)
    ax.plot(xs, ys, color=color, lw=2, marker="o", ms=5, zorder=3,
            label=f"α = {alpha:.1f}")
    ax.fill_between(xs, ys - yerrs, ys + yerrs, color=color, alpha=0.15, zorder=2)

ax.axhline(0.5, color="0.6", lw=0.8, ls="--", zorder=1, label="Chance")
ax.set_xlabel("Number of wrongly predicted concepts  ($n_{\\mathrm{wrong}}$)", fontsize=11)
ax.set_ylabel("Task accuracy", fontsize=11)
ax.set_title(
    "Error-set analysis: task accuracy vs concept errors\n"
    "High $\\alpha$ = task label leaked into embeddings",
    fontsize=11, fontweight="bold",
)
ax.set_xticks(range(K + 1))
ax.set_ylim(0.3, 1.05)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)
ax.tick_params(labelsize=9)
ax.legend(fontsize=9, loc="lower left", framealpha=0.85)

fig.suptitle(
    f"Controlled Gaussian validation  "
    f"(K={K} concepts, m={M} dims, N={N_TRAIN}+{N_TEST}, {N_SEEDS} seeds)",
    fontsize=10,
)

master_folder = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
PLOT_DIR = master_folder + "results/plots/cbm/"
os.makedirs(PLOT_DIR, exist_ok=True)
out = PLOT_DIR + "validate_error_set.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved → {out}")
plt.close(fig)
