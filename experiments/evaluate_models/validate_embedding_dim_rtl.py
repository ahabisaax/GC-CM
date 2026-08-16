"""
Embedding dimension sweep: RTL and RCL vs leakage strength, one line per m.

Mirrors validate_embedding_dim_paper.py (same generative model, same grid)
but uses RTL/RCL metrics instead of CVL/ICVL.

RTL panel: sweep alpha (task leakage), one curve per embedding dimension m.
RCL panel: sweep beta (inter-concept leakage), one curve per m.

Run from project root:
    python experiments/evaluate_models/validate_embedding_dim_rtl.py
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
    "lines.linewidth":    1.7,
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
# Parameters  (match validate_embedding_dim_paper.py)
# ---------------------------------------------------------------------------
N, K, N_TRAIN = 3000, 6, 2000
N_TEST  = N - N_TRAIN
N_SEEDS = 15
ALPHAS  = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])
DIMS    = [2, 4, 8, 16, 32, 64, 128]

# Colour + marker per dim (match validate_embedding_dim_paper.py)
STYLES = {
    2:   dict(color="#a50026", marker="o",  ls=(0, (1, 1)),    lw=1.5),
    4:   dict(color="#d73027", marker="v",  ls=(0, (3, 1)),    lw=1.5),
    8:   dict(color="#fc8d59", marker="^",  ls="--",           lw=1.6),
    16:  dict(color="#222222", marker="D",  ls="-",            lw=2.1),
    32:  dict(color="#4575b4", marker="s",  ls="-.",           lw=1.6),
    64:  dict(color="#313695", marker="P",  ls=(0, (5, 2)),    lw=1.5),
    128: dict(color="#74add1", marker="*",  ls=(0, (5,1,1,1)), lw=1.5),
}


# ---------------------------------------------------------------------------
# Data generator  (identical to validate_embedding_dim_paper.py)
# ---------------------------------------------------------------------------
def make_embeddings(m, alpha_task, alpha_ic, seed):
    rng_p    = np.random.RandomState(seed * 10_000 + m)
    mu_pos   = rng_p.randn(K, m)
    mu_neg   = rng_p.randn(K, m)
    task_sig = rng_p.randn(2, m)
    ic_sig_0 = rng_p.randn(m)
    ic_sig_1 = rng_p.randn(m)

    rng  = np.random.RandomState(seed)
    c    = rng.randint(0, 2, (N, K)).astype(float)
    y    = (c.sum(axis=1) >= K / 2).astype(int)

    c_hat = np.zeros((N, K, m))
    for k in range(K):
        base      = c[:, k:k+1] * mu_pos[k] + (1 - c[:, k:k+1]) * mu_neg[k]
        task_leak = alpha_task * task_sig[y]
        ic_leak   = np.zeros((N, m))
        if k == 1:
            ic_leak = alpha_ic * (c[:, 0:1] * ic_sig_1 + (1 - c[:, 0:1]) * ic_sig_0)
        c_hat[:, k, :] = base + task_leak + ic_leak + rng.randn(N, m)

    idx = rng.permutation(N)
    tr, te = idx[:N_TRAIN], idx[N_TRAIN:]
    return c_hat[tr], c_hat[te], c[tr], c[te], y[tr], y[te]


# ---------------------------------------------------------------------------
# Compute results
# ---------------------------------------------------------------------------
rtl_sum_means   = np.zeros((len(DIMS), len(ALPHAS)))
rtl_sum_stds    = np.zeros_like(rtl_sum_means)
rtl_norm_means  = np.zeros_like(rtl_sum_means)
rtl_norm_stds   = np.zeros_like(rtl_sum_means)
rcl_sum_means   = np.zeros_like(rtl_sum_means)
rcl_sum_stds    = np.zeros_like(rtl_sum_means)
rcl_norm_means  = np.zeros_like(rtl_sum_means)
rcl_norm_stds   = np.zeros_like(rtl_sum_means)

for di, m in enumerate(DIMS):
    print(f"m={m}")
    for ai, alpha in enumerate(ALPHAS):
        rtl_s_vals, rtl_n_vals = [], []
        rcl_s_vals, rcl_n_vals = [], []

        for seed in range(N_SEEDS):
            # RTL: task leakage only
            a = make_embeddings(m, alpha, 0.0, seed)
            r = compute_RTL_RCL(*a)
            rtl_s_vals.append(r["RTL_sum"])
            rtl_n_vals.append(r["RTL_norm"])

            # RCL: inter-concept leakage only
            b = make_embeddings(m, 0.0, alpha, seed)
            r2 = compute_RTL_RCL(*b)
            rcl_s_vals.append(r2["RCL_sum"])
            rcl_n_vals.append(r2["RCL_norm"])

        rtl_sum_means[di, ai]  = np.mean(rtl_s_vals)
        rtl_sum_stds[di, ai]   = np.std(rtl_s_vals)
        rtl_norm_means[di, ai] = np.mean(rtl_n_vals)
        rtl_norm_stds[di, ai]  = np.std(rtl_n_vals)
        rcl_sum_means[di, ai]  = np.mean(rcl_s_vals)
        rcl_sum_stds[di, ai]   = np.std(rcl_s_vals)
        rcl_norm_means[di, ai] = np.mean(rcl_n_vals)
        rcl_norm_stds[di, ai]  = np.std(rcl_n_vals)

        print(f"  α/β={alpha:.2f}  RTL_sum={rtl_sum_means[di,ai]:.4f}"
              f"  RTL_norm={rtl_norm_means[di,ai]:.4f}"
              f"  RCL_sum={rcl_sum_means[di,ai]:.4f}"
              f"  RCL_norm={rcl_norm_means[di,ai]:.4f}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PLOT_DIR = "results/plots/cbm/"
os.makedirs(PLOT_DIR, exist_ok=True)


def decorate(ax, xlabel, ylabel, xlim=None):
    ax.axhline(0, color="0.6", lw=0.7, ls=":", zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_ylim(bottom=-0.01)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))


def plot_lines(ax, means, stds):
    for di, m in enumerate(DIMS):
        s   = STYLES[m]
        lbl = f"$m={m}$" + (" (default)" if m == 16 else "")
        ax.plot(ALPHAS, means[di],
                color=s["color"], ls=s["ls"], lw=s["lw"],
                marker=s["marker"], ms=5, zorder=3, label=lbl)
        ax.fill_between(ALPHAS, means[di] - stds[di], means[di] + stds[di],
                        color=s["color"], alpha=0.06, zorder=2)


xlim = (ALPHAS[0] - 0.1, ALPHAS[-1] + 0.1)

# ===========================================================================
# FIGURE 1: RTL_sum
# ===========================================================================
fig1, ax1 = plt.subplots(figsize=(4.5, 3.2))
plot_lines(ax1, rtl_sum_means, rtl_sum_stds)
decorate(ax1, xlabel=r"Task-leakage strength $\alpha$", ylabel="RTL (sum)", xlim=xlim)
ax1.legend(loc="upper left", framealpha=0.92, edgecolor="0.78",
           ncol=1, handlelength=2.2, labelspacing=0.35)
fig1.tight_layout()
out1 = PLOT_DIR + "paper_rtl_dim_sweep_sum.pdf"
fig1.savefig(out1); fig1.savefig(out1.replace(".pdf", ".png"))
print(f"\nSaved → {out1}")
plt.close(fig1)

# ===========================================================================
# FIGURE 2: RCL_sum
# ===========================================================================
fig2, ax2 = plt.subplots(figsize=(4.5, 3.2))
plot_lines(ax2, rcl_sum_means, rcl_sum_stds)
decorate(ax2, xlabel=r"Inter-concept leakage strength $\beta$", ylabel="RCL (sum)", xlim=xlim)
ax2.legend(loc="upper left", framealpha=0.92, edgecolor="0.78",
           ncol=1, handlelength=2.2, labelspacing=0.35)
fig2.tight_layout()
out2 = PLOT_DIR + "paper_rcl_dim_sweep_sum.pdf"
fig2.savefig(out2); fig2.savefig(out2.replace(".pdf", ".png"))
print(f"Saved → {out2}")
plt.close(fig2)

# ===========================================================================
# FIGURE 3: Combined panel (RTL_sum | RCL_sum)
# ===========================================================================
fig3, axes3 = plt.subplots(1, 2, figsize=(6.8, 2.8))

panel_cfg = [
    (axes3[0], rtl_sum_means, rtl_sum_stds,
     r"Task-leakage strength $\alpha$", "RTL (sum)", "(a)"),
    (axes3[1], rcl_sum_means, rcl_sum_stds,
     r"Inter-concept leakage strength $\beta$", "RCL (sum)", "(b)"),
]

for ax, means, stds, xlabel, ylabel, letter in panel_cfg:
    for di, m in enumerate(DIMS):
        s = STYLES[m]
        ax.plot(ALPHAS, means[di],
                color=s["color"], ls=s["ls"], lw=s["lw"],
                marker="o", ms=3.5, zorder=3, label=f"$m={m}$")
        ax.fill_between(ALPHAS, means[di] - stds[di], means[di] + stds[di],
                        color=s["color"], alpha=0.08, zorder=2)
    ax.axhline(0, color="0.6", lw=0.7, ls=":", zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=-0.01)
    ax.text(-0.18, 1.13, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

handles, labels = axes3[0].get_legend_handles_labels()
fig3.legend(handles, labels,
            loc="lower center", ncol=len(DIMS),
            bbox_to_anchor=(0.5, -0.16),
            framealpha=0.9, edgecolor="0.75",
            fontsize=7.5, handlelength=1.8,
            title="Embedding dimension $m$",
            title_fontsize=7.5)

fig3.tight_layout(w_pad=1.8)
fig3.subplots_adjust(top=0.88, bottom=0.28)
out3 = PLOT_DIR + "paper_rtl_rcl_dim_sweep_combined.pdf"
fig3.savefig(out3); fig3.savefig(out3.replace(".pdf", ".png"))
print(f"Saved combined → {out3}")
plt.close(fig3)

# ===========================================================================
# FIGURE 4: RTL_norm
# ===========================================================================
fig4, ax4 = plt.subplots(figsize=(4.5, 3.2))
plot_lines(ax4, rtl_norm_means, rtl_norm_stds)
decorate(ax4, xlabel=r"Task-leakage strength $\alpha$", ylabel="RTL (norm)", xlim=xlim)
ax4.legend(loc="upper left", framealpha=0.92, edgecolor="0.78",
           ncol=1, handlelength=2.2, labelspacing=0.35)
fig4.tight_layout()
out4 = PLOT_DIR + "paper_rtl_dim_sweep_norm.pdf"
fig4.savefig(out4); fig4.savefig(out4.replace(".pdf", ".png"))
print(f"\nSaved → {out4}")
plt.close(fig4)

# ===========================================================================
# FIGURE 5: RCL_norm
# ===========================================================================
fig5, ax5 = plt.subplots(figsize=(4.5, 3.2))
plot_lines(ax5, rcl_norm_means, rcl_norm_stds)
decorate(ax5, xlabel=r"Inter-concept leakage strength $\beta$", ylabel="RCL (norm)", xlim=xlim)
ax5.legend(loc="upper left", framealpha=0.92, edgecolor="0.78",
           ncol=1, handlelength=2.2, labelspacing=0.35)
fig5.tight_layout()
out5 = PLOT_DIR + "paper_rcl_dim_sweep_norm.pdf"
fig5.savefig(out5); fig5.savefig(out5.replace(".pdf", ".png"))
print(f"Saved → {out5}")
plt.close(fig5)

# ===========================================================================
# FIGURE 6: Combined panel norm (RTL_norm | RCL_norm)
# ===========================================================================
fig6, axes6 = plt.subplots(1, 2, figsize=(6.8, 2.8))

panel_cfg_norm = [
    (axes6[0], rtl_norm_means, rtl_norm_stds,
     r"Task-leakage strength $\alpha$", "RTL (norm)", "(a)"),
    (axes6[1], rcl_norm_means, rcl_norm_stds,
     r"Inter-concept leakage strength $\beta$", "RCL (norm)", "(b)"),
]

for ax, means, stds, xlabel, ylabel, letter in panel_cfg_norm:
    for di, m in enumerate(DIMS):
        s = STYLES[m]
        ax.plot(ALPHAS, means[di],
                color=s["color"], ls=s["ls"], lw=s["lw"],
                marker="o", ms=3.5, zorder=3, label=f"$m={m}$")
        ax.fill_between(ALPHAS, means[di] - stds[di], means[di] + stds[di],
                        color=s["color"], alpha=0.08, zorder=2)
    ax.axhline(0, color="0.6", lw=0.7, ls=":", zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(bottom=-0.01)
    ax.text(-0.18, 1.13, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

handles, labels = axes6[0].get_legend_handles_labels()
fig6.legend(handles, labels,
            loc="lower center", ncol=len(DIMS),
            bbox_to_anchor=(0.5, -0.16),
            framealpha=0.9, edgecolor="0.75",
            fontsize=7.5, handlelength=1.8,
            title="Embedding dimension $m$",
            title_fontsize=7.5)

fig6.tight_layout(w_pad=1.8)
fig6.subplots_adjust(top=0.88, bottom=0.28)
out6 = PLOT_DIR + "paper_rtl_rcl_dim_sweep_norm_combined.pdf"
fig6.savefig(out6); fig6.savefig(out6.replace(".pdf", ".png"))
print(f"Saved combined norm → {out6}")
plt.close(fig6)

print("Done.")
