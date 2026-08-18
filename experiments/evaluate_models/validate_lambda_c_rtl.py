"""
Controlled Gaussian validation: RTL and RCL vs concept supervision weight λ_c.

Mirrors validate_lambda_c.py (same generative model) but uses RTL/RCL metrics.

Motivation
----------
Higher λ_c → concept supervision dominates → embeddings stay anchored →
less task leakage (RTL↓) and less inter-concept leakage (RCL↓).
We model this as effective leakage strength α_eff = α / λ_c.

Run from project root:
    python experiments/evaluate_models/validate_lambda_c_rtl.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from xai_concept_leakage.metrics.leakage import compute_RTL_RCL

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "axes.linewidth":     0.75,
    "lines.linewidth":    1.8,
    "lines.markersize":   5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.22,
    "grid.linewidth":     0.5,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

# ---------------------------------------------------------------------------
# Parameters  (match validate_lambda_c.py)
# ---------------------------------------------------------------------------
N, K, M   = 3000, 6, 16
N_TRAIN   = 2000
N_TEST    = N - N_TRAIN
N_SEEDS   = 20
ALPHA_BASE = 1.5

LAMBDA_C        = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
LAM_EXPERIMENTAL = {0.1, 0.5, 1.0}

C_RTL  = "#0072B2"
C_RCL  = "#D55E00"
C_NORM = "#009E73"

# Fixed prototypes (same seed as validate_lambda_c.py)
_rng     = np.random.RandomState(0)
MU_POS   = _rng.randn(K, M)
MU_NEG   = _rng.randn(K, M)
TASK_SIG = _rng.randn(2, M)
IC_SIG_0 = _rng.randn(M)
IC_SIG_1 = _rng.randn(M)


# ---------------------------------------------------------------------------
# Data generator  (identical to validate_lambda_c.py)
# ---------------------------------------------------------------------------
def make_embeddings(alpha_task, alpha_ic, seed):
    rng = np.random.RandomState(seed)
    c   = rng.randint(0, 2, (N, K)).astype(float)
    y   = (c.sum(axis=1) >= K / 2).astype(int)

    c_hat = np.zeros((N, K, M))
    for k in range(K):
        base      = c[:, k:k+1] * MU_POS[k] + (1 - c[:, k:k+1]) * MU_NEG[k]
        task_leak = alpha_task * TASK_SIG[y]
        ic_leak   = np.zeros((N, M))
        if k == 1:
            ic_leak = alpha_ic * (c[:, 0:1] * IC_SIG_1 + (1 - c[:, 0:1]) * IC_SIG_0)
        c_hat[:, k, :] = base + task_leak + ic_leak + rng.randn(N, M)

    idx = rng.permutation(N)
    tr, te = idx[:N_TRAIN], idx[N_TRAIN:]
    return c_hat[tr], c_hat[te], c[tr], c[te], y[tr], y[te]


# ---------------------------------------------------------------------------
# Sweep λ_c
# ---------------------------------------------------------------------------
print(f"=== RTL and RCL vs λ_c  (base leakage pressure α=β={ALPHA_BASE}) ===")
print(f"    Effective leakage = {ALPHA_BASE} / λ_c\n")

rtl_sum_means,  rtl_sum_stds  = [], []
rtl_norm_means, rtl_norm_stds = [], []
rcl_sum_means,  rcl_sum_stds  = [], []
rcl_norm_means, rcl_norm_stds = [], []
rtl_gm_means,   rtl_gm_stds   = [], []
rcl_gm_means,   rcl_gm_stds   = [], []

for lam in LAMBDA_C:
    alpha_eff = ALPHA_BASE / lam

    rtl_s_vals, rtl_n_vals, rtl_gm_vals = [], [], []
    rcl_s_vals, rcl_n_vals, rcl_gm_vals = [], [], []

    for seed in range(N_SEEDS):
        # RTL: task leakage only
        args = make_embeddings(alpha_eff, 0.0, seed)
        r = compute_RTL_RCL(*args)
        rtl_s_vals.append(r["RTL_sum"])
        rtl_n_vals.append(r["RTL_norm"])
        rg = compute_RTL_RCL(*args, global_norm=True)
        rtl_gm_vals.append(rg["RTL_sum"] / M)

        # RCL: inter-concept leakage only
        args_ic = make_embeddings(0.0, alpha_eff, seed)
        r_ic = compute_RTL_RCL(*args_ic)
        rcl_s_vals.append(r_ic["RCL_sum"])
        rcl_n_vals.append(r_ic["RCL_norm"])
        rg_ic = compute_RTL_RCL(*args_ic, global_norm=True)
        rcl_gm_vals.append(rg_ic["RCL_sum"] / M)

    rtl_sum_means.append(np.mean(rtl_s_vals));  rtl_sum_stds.append(np.std(rtl_s_vals))
    rtl_norm_means.append(np.mean(rtl_n_vals)); rtl_norm_stds.append(np.std(rtl_n_vals))
    rcl_sum_means.append(np.mean(rcl_s_vals));  rcl_sum_stds.append(np.std(rcl_s_vals))
    rcl_norm_means.append(np.mean(rcl_n_vals)); rcl_norm_stds.append(np.std(rcl_n_vals))
    rtl_gm_means.append(np.mean(rtl_gm_vals));  rtl_gm_stds.append(np.std(rtl_gm_vals))
    rcl_gm_means.append(np.mean(rcl_gm_vals));  rcl_gm_stds.append(np.std(rcl_gm_vals))

    print(f"  λ_c={lam:.2f}  α_eff={alpha_eff:.2f}"
          f"  RTL_sum={rtl_sum_means[-1]:.4f}±{rtl_sum_stds[-1]:.4f}"
          f"  RTL_norm={rtl_norm_means[-1]:.4f}"
          f"  RTL/m(gl)={rtl_gm_means[-1]:.4f}"
          f"  RCL_sum={rcl_sum_means[-1]:.4f}±{rcl_sum_stds[-1]:.4f}"
          f"  RCL_norm={rcl_norm_means[-1]:.4f}"
          f"  RCL/m(gl)={rcl_gm_means[-1]:.4f}")

rtl_sum_means  = np.array(rtl_sum_means);  rtl_sum_stds  = np.array(rtl_sum_stds)
rtl_norm_means = np.array(rtl_norm_means); rtl_norm_stds = np.array(rtl_norm_stds)
rcl_sum_means  = np.array(rcl_sum_means);  rcl_sum_stds  = np.array(rcl_sum_stds)
rcl_norm_means = np.array(rcl_norm_means); rcl_norm_stds = np.array(rcl_norm_stds)
rtl_gm_means   = np.array(rtl_gm_means);  rtl_gm_stds   = np.array(rtl_gm_stds)
rcl_gm_means   = np.array(rcl_gm_means);  rcl_gm_stds   = np.array(rcl_gm_stds)


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------
PLOT_DIR = "results/plots/cbm/"
os.makedirs(PLOT_DIR, exist_ok=True)


def _decorate_lam(ax, ylabel, color, s_means):
    ax.set_xscale("log")
    ax.set_xlabel(r"Concept supervision weight $\lambda_c$")
    ax.set_ylabel(ylabel, color=color)
    ax.tick_params(axis="y", colors=color)
    ax.set_xlim(LAMBDA_C[0] * 0.6, LAMBDA_C[-1] * 1.8)
    ax.set_ylim(bottom=0)
    ax.set_xticks(LAMBDA_C)
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.tick_params(axis="x", which="minor", bottom=False)


def _single_panel(ax, means, stds, color, ylabel, letter):
    ax.plot(LAMBDA_C, means, color=color, lw=1.9, marker="o", ms=5, zorder=3)
    ax.fill_between(LAMBDA_C, means - stds, means + stds,
                    color=color, alpha=0.15, zorder=2)
    _decorate_lam(ax, ylabel, color, means)
    ax.text(-0.18, 1.10, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")


# ---------------------------------------------------------------------------
# Figure 1: Combined twin-axes (RTL | RCL), sum left / norm right
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

panel_data = [
    (axes[0], rtl_sum_means, rtl_sum_stds, rtl_norm_means, rtl_norm_stds,
     C_RTL, "RTL (sum)", "RTL (norm)", "(a)"),
    (axes[1], rcl_sum_means, rcl_sum_stds, rcl_norm_means, rcl_norm_stds,
     C_RCL, "RCL (sum)", "RCL (norm)", "(b)"),
]

for ax, s_means, s_stds, n_means, n_stds, color, ylabel_s, ylabel_n, letter in panel_data:
    ax2 = ax.twinx()
    l1, = ax.plot(LAMBDA_C, s_means, color=color, lw=1.9, marker="o", ms=5,
                  zorder=3, label="sum")
    ax.fill_between(LAMBDA_C, s_means - s_stds, s_means + s_stds,
                    color=color, alpha=0.15, zorder=2)
    l2, = ax2.plot(LAMBDA_C, n_means, color=C_NORM, lw=1.7, marker="s", ms=5,
                   ls="--", zorder=3, label="norm")
    ax2.fill_between(LAMBDA_C, n_means - n_stds, n_means + n_stds,
                     color=C_NORM, alpha=0.12, zorder=2)
    _decorate_lam(ax, ylabel_s, color, s_means)
    ax2.set_ylabel(ylabel_n, color=C_NORM)
    ax2.tick_params(axis="y", colors=C_NORM)
    ax2.set_ylim(bottom=0)
    ax.legend(handles=[l1, l2], loc="upper right", fontsize=8)
    ax.text(-0.18, 1.10, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

fig.tight_layout(w_pad=1.8)
fig.subplots_adjust(top=0.88)
out = PLOT_DIR + "paper_lambda_c_rtl_rcl.pdf"
fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"))
print(f"\nSaved → {out}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: Sum only (RTL_sum | RCL_sum)
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(6.8, 2.8))
_single_panel(axes2[0], rtl_sum_means, rtl_sum_stds, C_RTL, "RTL (sum)", "(a)")
_single_panel(axes2[1], rcl_sum_means, rcl_sum_stds, C_RCL, "RCL (sum)", "(b)")
fig2.tight_layout(w_pad=1.8)
fig2.subplots_adjust(top=0.88)
out2 = PLOT_DIR + "paper_lambda_c_rtl_rcl_sum.pdf"
fig2.savefig(out2); fig2.savefig(out2.replace(".pdf", ".png"))
print(f"Saved → {out2}")
plt.close(fig2)

# ---------------------------------------------------------------------------
# Figure 3: Norm only (RTL_norm | RCL_norm)
# ---------------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 2, figsize=(6.8, 2.8))
_single_panel(axes3[0], rtl_norm_means, rtl_norm_stds, C_RTL, "RTL (norm)", "(a)")
_single_panel(axes3[1], rcl_norm_means, rcl_norm_stds, C_RCL, "RCL (norm)", "(b)")
fig3.tight_layout(w_pad=1.8)
fig3.subplots_adjust(top=0.88)
out3 = PLOT_DIR + "paper_lambda_c_rtl_rcl_norm.pdf"
fig3.savefig(out3); fig3.savefig(out3.replace(".pdf", ".png"))
print(f"Saved → {out3}")
plt.close(fig3)

# ---------------------------------------------------------------------------
# Figure 4: RTL/m and RCL/m (global norm)
# ---------------------------------------------------------------------------
fig4, axes4 = plt.subplots(1, 2, figsize=(6.8, 2.8))
_single_panel(axes4[0], rtl_gm_means, rtl_gm_stds, C_RTL, "RTL", "(a)")
_single_panel(axes4[1], rcl_gm_means, rcl_gm_stds, C_RCL, "RCL", "(b)")
fig4.tight_layout(w_pad=1.8)
fig4.subplots_adjust(top=0.88)
out4 = PLOT_DIR + "paper_lambda_c_rtl_rcl_global_m.pdf"
fig4.savefig(out4); fig4.savefig(out4.replace(".pdf", ".png"))
print(f"Saved → {out4}")
plt.close(fig4)

print("Done.")
