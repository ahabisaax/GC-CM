"""
Pareto scatter plot: Task Accuracy vs CTL across CBM models.
One subplot per dataset. Pareto frontier drawn per subplot.

Run from project root:
  python experiments/evaluate_models/plot_pareto_cbm.py
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import joblib

master_folder = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
R = master_folder + "results/"

# ---------------------------------------------------------------------------
# Wong palette
# ---------------------------------------------------------------------------
C = {
    "Joint CBM":    "#0072B2",
    "Hard CBM":     "#009E73",
    "Sequential CBM": "#E69F00",
    "GC-CBM":       "#D55E00",
}
MARKERS = {
    "Joint CBM":    "o",
    "Hard CBM":     "s",
    "Sequential CBM": "D",
    "GC-CBM":       "^",
}
SIZES = {
    "Joint CBM":    70,
    "Hard CBM":     90,
    "Sequential CBM": 90,
    "GC-CBM":       90,
}

def lstr(l): return "1" if l == 1.0 else str(l)

LAM = [0.1, 0.5, 1.0]

SOURCES = {
    "TabularToy": {
        "Joint CBM":      {l: (R+"tabulartoy_25_10k_models",
                               "SoftCBM_adam_lr0.05_bs64_lam_c"+lstr(l)) for l in LAM},
        "Hard CBM":       {1.0:(R+"tabulartoy_25_10k_models",
                               "HardCBM_adam_lr0.05_bs64_lam_c1")},
        "Sequential CBM": {1.0:(R+"tabulartoy_25_10k_models_acbm_shared_critic",
                               "SeqCBM_adam_lr0.05_bs64_lam_c1")},
        "GC-CBM":         {l: (R+"tabulartoy_25_10k_models_acbm_shared_critic",
                               "CRCBM_adam_lr0.01_bs64_lam1_none_lam_c"+lstr(l)+"_shared_critic") for l in LAM},
    },
    "dSprites": {
        "Joint CBM":      {l: (R+"dsprites_dep_0_models",
                               "SoftCBM_adam_lr1e-04_bs64_lam_c"+lstr(l)) for l in LAM},
        "Hard CBM":       {1.0:(R+"dsprites_dep_0_models",
                               "HardCBM_adam_lr1e-04_bs64_lam_c1")},
        "Sequential CBM": {1.0:(R+"dsprites_sequential_acbm",
                               "SeqCBM_adam_lr1e-04_bs64_lam_c1")},
        "GC-CBM":         {l: (R+"dsprites_ACBM_shared_critic",
                               "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c"+lstr(l)+"_shared_critic") for l in LAM},
    },
    "CUB-200": {
        "Joint CBM":      {l: (R+"cub_soft_0.01",
                               "SoftCBM_adam_lr1e-04_bs256_lam_c"+lstr(l)) for l in LAM},
        "Hard CBM":       {1.0:(R+"cub_soft_0.01",
                               "HardCBM_adam_lr1e-04_bs256_lam_c1")},
        "Sequential CBM": {1.0:(R+"cub_acbm_shared_critic",
                               "SeqCBM_adam_lr1e-04_bs256_lam_c1")},
        "GC-CBM":         {l: (R+"cub_acbm_shared_critic",
                               "ACBM_adam_lr1e-04_bs256_lam1_none_lam_c"+lstr(l)+"_shared_critic") for l in LAM},
    },
}

DATASET_ORDER = ["TabularToy", "dSprites", "CUB-200"]
MODEL_ORDER   = ["Joint CBM", "Hard CBM", "Sequential CBM", "GC-CBM"]


def load(folder, mdir):
    path = os.path.join(folder, mdir)
    splits = sorted(glob.glob(os.path.join(path, "*_split_*_results.joblib")))
    task, ctl = [], []
    for s in splits:
        d = joblib.load(s)
        task.append(d.get("test_acc_y", np.nan))
        ctl.append(d.get("test_ctl_average", np.nan))
    t = np.array(task); c = np.array(ctl)
    t = t[~np.isnan(t)]; c = c[~np.isnan(c)]
    n = min(len(t), len(c))
    return np.mean(t[:n])*100, np.std(t[:n])*100, np.mean(c[:n]), np.std(c[:n])


def pareto_frontier(points):
    """Return indices of Pareto-optimal points (max task acc, min CTL)."""
    pts = np.array(points)   # shape (N, 2): col0=task_acc, col1=CTL
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if higher task AND lower CTL
            if pts[j, 0] >= pts[i, 0] and pts[j, 1] <= pts[i, 1]:
                if pts[j, 0] > pts[i, 0] or pts[j, 1] < pts[i, 1]:
                    is_pareto[i] = False
                    break
    return np.where(is_pareto)[0]


# ---------------------------------------------------------------------------
# Build data
# ---------------------------------------------------------------------------
# points[ds][model] = list of (task_mean, task_std, ctl_mean, ctl_std, lam_label)
points = {ds: {m: [] for m in MODEL_ORDER} for ds in DATASET_ORDER}

for ds in DATASET_ORDER:
    for model in MODEL_ORDER:
        for lam, (folder, mdir) in SOURCES[ds][model].items():
            try:
                tm, ts, cm, cs = load(folder, mdir)
                lbl = f"λ={lam}" if lam != 1.0 else "λ=1"
                points[ds][model].append((tm, ts, cm, cs, lbl))
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

for col, ds in enumerate(DATASET_ORDER):
    ax = axes[col]

    all_pts = []   # (task_mean, ctl_mean) for Pareto calculation

    for model in MODEL_ORDER:
        pts = points[ds][model]
        if not pts:
            continue

        xs = np.array([p[0] for p in pts])   # task acc
        xe = np.array([p[1] for p in pts])   # task std
        ys = np.array([p[2] for p in pts])   # CTL
        ye = np.array([p[3] for p in pts])   # CTL std
        lbls = [p[4] for p in pts]

        # Sort by lambda for line joining
        order = np.argsort([float(l.split("=")[1]) for l in lbls])
        xs, xe, ys, ye = xs[order], xe[order], ys[order], ye[order]
        lbls = [lbls[i] for i in order]

        ax.scatter(ys, xs, color=C[model], marker=MARKERS[model],
                   s=SIZES[model], label=model, zorder=5,
                   edgecolors="white", linewidths=0.7)

        # Lambda annotations
        for x, y, lbl in zip(xs, ys, lbls):
            ax.annotate(lbl, (y, x),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=6.5, color=C[model], alpha=0.85)

        all_pts.extend(zip(xs, ys))

    # Pareto frontier — smooth line through non-dominated points
    # (max task acc = up, min CTL = left → frontier goes top-left to bottom-right)
    if len(all_pts) >= 2:
        idx = pareto_frontier(all_pts)
        pf  = np.array([all_pts[i] for i in idx])
        pf  = pf[np.argsort(pf[:, 1])]   # sort by CTL ascending
        ax.plot(pf[:, 1], pf[:, 0],
                color="0.25", lw=1.4, ls="--", zorder=4, alpha=0.75,
                label="Pareto frontier")

    ax.set_xlabel("CTL", fontsize=11)
    if col == 0:
        ax.set_ylabel("Task accuracy (%)", fontsize=11)
    ax.set_title(ds, fontsize=12, fontweight="bold", pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.6)
    ax.xaxis.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    if col == 0:
        ax.legend(fontsize=8.5, loc="upper right", framealpha=0.9,
                  frameon=True, edgecolor="0.8", ncol=1)

fig.suptitle("CTL vs Task accuracy — CBM model family", fontsize=13, fontweight="bold")

PLOT_DIR = master_folder + "results/plots/cbm/"
os.makedirs(PLOT_DIR, exist_ok=True)
out = PLOT_DIR + "pareto_task_ctl_cbm.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig)
