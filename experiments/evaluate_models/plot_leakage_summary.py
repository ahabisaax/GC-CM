"""
Cross-dataset leakage summary figure.
Loads results_*_suite.dict files and produces a multi-panel summary.

Run from project root:
    python experiments/evaluate_models/plot_leakage_summary.py
"""
import os, sys
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

master = os.getcwd().rstrip("/") + "/"
sys.path.insert(0, master)

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
res_tt  = joblib.load(master + "results/results_tabulartoy_suite.dict")
res_ds  = joblib.load(master + "results/results_dsprites_suite.dict")
res_cub = joblib.load(master + "results/results_cub_suite.dict")

FOLDS = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
C_CEM   = "#4878CF"
C_CRCEM = "#D65F5F"
ALPHA_BAR = 0.85

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def collect(res, fn):
    cem   = [fn(res["cem"][f])   for f in FOLDS if f in res["cem"]]
    crcem = [fn(res["crcem"][f]) for f in FOLDS if f in res["crcem"]]
    return (np.mean(cem),   np.std(cem,   ddof=1),
            np.mean(crcem), np.std(crcem, ddof=1))

def err_acc_fn(r):
    vals = [v["task_acc"] for k, v in r["error_set"].items() if k > 0]
    return np.mean(vals) if vals else np.nan

def icl_delta_fn(r):
    K    = r["icl_probe"]["acc_diff"].shape[0]
    mask = ~np.eye(K, dtype=bool)
    return r["icl_probe"]["acc_diff"][mask].mean()

def probe_avg_fn(r):
    return np.array(r["probe_acc"]).mean(axis=0)   # (n_comp,)

def error_curve(r, fold, max_k=None):
    es = r[fold]["error_set"]
    ks = sorted(es.keys())
    if max_k is not None:
        ks = [k for k in ks if k <= max_k]
    x = np.array(ks)
    y = np.array([es[k]["task_acc"] for k in ks])
    n = np.array([es[k]["n_samples"] for k in ks])
    return x, y, n

# ---------------------------------------------------------------------------
# Pre-compute stats
# ---------------------------------------------------------------------------
datasets   = ["TabularToy", "dSprites", "CUB"]
res_list   = [res_tt, res_ds, res_cub]

task_stats = [collect(r, lambda r: r["task_acc"])  for r in res_list]
cacc_stats = [collect(r, lambda r: r["c_acc"])     for r in res_list]
err_stats  = [collect(r, err_acc_fn)               for r in res_list]
icld_stats = [collect(r, icl_delta_fn)             for r in res_list]

# KSG metrics — TabularToy and dSprites only
ctl_stats  = [collect(r, lambda r: np.mean(r["ctl_mean"])) for r in [res_tt, res_ds]]
icl_stats  = [collect(r, lambda r: r["icl_mean"])           for r in [res_tt, res_ds]]
cvl_stats  = [collect(r, lambda r: r["cvl"]["CVL"])         for r in [res_tt, res_ds]]

# Probe curves averaged over concepts — fold_1 representative
probe_cem_tt    = probe_avg_fn(res_tt["cem"]["fold_1"])
probe_crcem_tt  = probe_avg_fn(res_tt["crcem"]["fold_1"])
probe_cem_ds    = probe_avg_fn(res_ds["cem"]["fold_1"])
probe_crcem_ds  = probe_avg_fn(res_ds["crcem"]["fold_1"])
probe_cem_cub   = probe_avg_fn(res_cub["cem"]["fold_1"])
probe_crcem_cub = probe_avg_fn(res_cub["crcem"]["fold_1"])

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(
    3, 5,
    figure=fig,
    hspace=0.52,
    wspace=0.40,
    left=0.06, right=0.98,
    top=0.93,  bottom=0.06,
)

ax_task  = fig.add_subplot(gs[0, 0])
ax_cacc  = fig.add_subplot(gs[0, 1])
ax_err   = fig.add_subplot(gs[0, 2])
ax_icld  = fig.add_subplot(gs[0, 3])
ax_ksg   = fig.add_subplot(gs[0, 4])

ax_ec_tt  = fig.add_subplot(gs[1, 0])
ax_ec_ds  = fig.add_subplot(gs[1, 1:3])
ax_ec_cub = fig.add_subplot(gs[1, 3:5])

ax_pr_tt  = fig.add_subplot(gs[2, 0:2])
ax_pr_ds  = fig.add_subplot(gs[2, 2:4])
ax_pr_cub = fig.add_subplot(gs[2, 4])

# ---------------------------------------------------------------------------
# Helper: grouped bar for 3 datasets
# ---------------------------------------------------------------------------
def grouped_bars(ax, stats, ylabel, title, pct=False, ylim=None, ds_labels=None):
    labels = ds_labels or datasets
    x      = np.arange(len(labels))
    w      = 0.33
    cem_m,   cem_s,   crcem_m,   crcem_s   = zip(*stats)
    bars1 = ax.bar(x - w/2, cem_m,   w, yerr=cem_s,   capsize=3,
                   color=C_CEM,   alpha=ALPHA_BAR, label="CEM",   error_kw={"lw":1.2})
    bars2 = ax.bar(x + w/2, crcem_m, w, yerr=crcem_s, capsize=3,
                   color=C_CRCEM, alpha=ALPHA_BAR, label="CRCEM", error_kw={"lw":1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title,   fontsize=9.5, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    if pct:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

# ---------------------------------------------------------------------------
# Row 0 — bar charts
# ---------------------------------------------------------------------------
grouped_bars(ax_task, task_stats,  "Accuracy",      "Task accuracy",
             ylim=(0.55, 0.82))
grouped_bars(ax_cacc, cacc_stats,  "Accuracy",      "All-concept accuracy",
             ylim=(0, 0.42))
grouped_bars(ax_err,  err_stats,   "Task acc (k>0)", "Error-set task acc\n(unweighted mean, k>0 wrong)",
             ylim=(0, 0.52))
grouped_bars(ax_icld, icld_stats,  "Δacc",          "ICL probe Δacc\n(mean off-diagonal)",
             ylim=(0, 0.08))

# KSG metrics — TabularToy + dSprites only
def ksg_bars(ax):
    ds2 = ["TabularToy", "dSprites"]
    x   = np.arange(3)       # CTL, ICL, CVL
    w   = 0.18
    grp_labels = ["CTL", "ICL\n(KSG)", "CVL"]
    for di, (ds, cs, vs) in enumerate(zip(ctl_stats, icl_stats, cvl_stats)):
        offsets = np.array([-w, 0, w]) + di*0.06
        for gi, (stat, lbl) in enumerate(zip([cs, ds, vs], grp_labels)):
            m_cem, s_cem, m_cr, s_cr = stat
            col = C_CEM if di == 0 else C_CRCEM
            pass

    # Simpler: one group per metric, two bars per group (CEM, CRCEM)
    # Use TabularToy + dSprites interleaved
    xtick_labels = []
    pos = 0
    tick_pos = []
    for gi, (grp, ctl, icl, cvl) in enumerate(
            zip(["CTL","ICL\n(KSG)","CVL"],
                [ctl_stats, icl_stats, cvl_stats],
                [ctl_stats, icl_stats, cvl_stats],
                [ctl_stats, icl_stats, cvl_stats])):
        pass

    # Cleaner: 3 metric groups × 2 datasets × 2 models
    stats3 = [ctl_stats, icl_stats, cvl_stats]
    labels3 = ["CTL", "ICL (KSG)", "CVL"]
    xs = np.array([0, 1, 2])
    w  = 0.20
    offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]   # TT-CEM, TT-CRCEM, DS-CEM, DS-CRCEM
    cols    = [C_CEM, C_CRCEM, C_CEM, C_CRCEM]
    hatch   = ["", "", "///", "///"]
    for metric_idx, stat in enumerate(stats3):
        for di, (ds_stat, off, col, ht) in enumerate(
                zip([stat[0], stat[0], stat[1], stat[1]], offsets, cols, hatch)):
            m_cem, s_cem, m_cr, s_cr = stat[int(di >= 2)]
            m = m_cem if di % 2 == 0 else m_cr
            s = s_cem if di % 2 == 0 else s_cr
            ax.bar(metric_idx + off, m, w, yerr=s, capsize=2,
                   color=col, alpha=ALPHA_BAR, hatch=ht,
                   error_kw={"lw": 1.0})

    ax.set_xticks(xs)
    ax.set_xticklabels(labels3, fontsize=8.5)
    ax.set_title("KSG leakage metrics\n(TabularToy  ■   dSprites ▨)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("MI (nats) / R²", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)

ksg_bars(ax_ksg)

# Legend on ax_ksg
from matplotlib.patches import Patch
legend_els = [
    Patch(fc=C_CEM,   alpha=ALPHA_BAR, label="CEM"),
    Patch(fc=C_CRCEM, alpha=ALPHA_BAR, label="CRCEM"),
    Patch(fc="grey",  alpha=0.4, hatch="",    label="TabularToy"),
    Patch(fc="grey",  alpha=0.4, hatch="///", label="dSprites"),
]
ax_ksg.legend(handles=legend_els, fontsize=7, loc="upper right",
              framealpha=0.7, ncol=2)

# Add legend to row-0 bar charts
for ax in [ax_task, ax_cacc, ax_err, ax_icld]:
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.7)

# ---------------------------------------------------------------------------
# Row 1 — error-set curves
# ---------------------------------------------------------------------------
def plot_error_set(ax, res_cem, res_crcem, max_k, title, xlabel=True):
    for label, r_dict, col in [
        ("CEM",   res_cem,   C_CEM),
        ("CRCEM", res_crcem, C_CRCEM),
    ]:
        x, y, n = error_curve(r_dict, "fold_1", max_k=max_k)
        mask = n >= 3
        ax.plot(x[mask], y[mask], "o-", color=col, lw=1.6,
                ms=3.5, label=label, alpha=0.9)
        ax.fill_between(x[mask], y[mask], alpha=0.08, color=col)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(left=0)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Task accuracy", fontsize=8.5)
    if xlabel:
        ax.set_xlabel("# wrong concepts (k)", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.7)
    ax.tick_params(labelsize=8)
    ax.axhline(0, color="grey", lw=0.6, ls="--", alpha=0.5)

# TabularToy (max k=3)
plot_error_set(ax_ec_tt,  res_tt["cem"],  res_tt["crcem"],  max_k=3,
               title="Error-set — TabularToy\n(3 concepts)")
ax_ec_tt.set_xlim(-0.2, 3.3)

# dSprites (max k=5)
plot_error_set(ax_ec_ds,  res_ds["cem"],  res_ds["crcem"],  max_k=5,
               title="Error-set — dSprites\n(5 concepts)")
ax_ec_ds.set_xlim(-0.2, 5.3)

# CUB (max k=30, drop small bins)
plot_error_set(ax_ec_cub, res_cub["cem"], res_cub["crcem"], max_k=30,
               title="Error-set — CUB-200\n(112 concepts, fold_1, n≥3)")
ax_ec_cub.set_xlabel("# wrong concepts (k)", fontsize=8.5)

# ---------------------------------------------------------------------------
# Row 2 — probe accuracy curves
# ---------------------------------------------------------------------------
def plot_probe(ax, cem_avg, crcem_avg, title, max_comp=None):
    n = min(len(cem_avg), len(crcem_avg))
    if max_comp:
        n = min(n, max_comp)
    xs = np.arange(1, n + 1)
    ax.plot(xs, cem_avg[:n],   "-", color=C_CEM,   lw=1.8, label="CEM",   alpha=0.9)
    ax.plot(xs, crcem_avg[:n], "-", color=C_CRCEM, lw=1.8, label="CRCEM", alpha=0.9)
    ax.fill_between(xs, cem_avg[:n],   alpha=0.10, color=C_CEM)
    ax.fill_between(xs, crcem_avg[:n], alpha=0.10, color=C_CRCEM)
    ax.set_xlabel("PCA components", fontsize=8.5)
    ax.set_ylabel("Probe accuracy (task)", fontsize=8.5)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_xlim(1, n)

plot_probe(ax_pr_tt,  probe_cem_tt,  probe_crcem_tt,
           "Probe accuracy vs PCA — TabularToy\n(avg over 3 concepts, fold_1)")
plot_probe(ax_pr_ds,  probe_cem_ds,  probe_crcem_ds,
           "Probe accuracy vs PCA — dSprites\n(avg over 5 concepts, fold_1)")
plot_probe(ax_pr_cub, probe_cem_cub, probe_crcem_cub,
           "Probe — CUB\n(avg 112 concepts)", max_comp=32)

# Add chance line for CUB (1/200)
ax_pr_cub.axhline(1/200, color="grey", lw=1, ls="--", alpha=0.6, label="chance (1/200)")
ax_pr_cub.legend(fontsize=6.5, loc="upper right", framealpha=0.7)

# ---------------------------------------------------------------------------
# Global title and save
# ---------------------------------------------------------------------------
fig.suptitle(
    "Leakage metrics: CEM vs CRCEM across datasets",
    fontsize=13, fontweight="bold", y=0.97,
)

out_path = master + "results/plots/leakage_summary.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {out_path}")
plt.close(fig)
