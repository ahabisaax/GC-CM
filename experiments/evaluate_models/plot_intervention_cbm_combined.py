"""
Combined CBM intervention plot — TabularToy | dSprites | CUB-200 side by side.
Shared legend below the subplots.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"

LAMS      = ["lam_c0.1", "lam_c0.5", "lam_c1"]
LAM_LABEL = [r"$\lambda_c=0.1$", r"$\lambda_c=0.5$", r"$\lambda_c=1.0$"]

C_HARD  = "#2c2c2c"
C_SEQ   = "#7f7f7f"
C_JOINT = ["#f4a896", "#e05c3a", "#8b1a00"]
C_GCCBM = ["#92c4de", "#2b7ab8", "#003f7d"]


def get_curves(suite, model_key, lam_key):
    curves = []
    for fold_r in suite.get(model_key, {}).get(lam_key, {}).values():
        iv = fold_r.get("interv", {}).get("random", [])
        if iv and len(iv[0]) > 0:
            curves.append(np.mean(iv, axis=0))
    return np.array(curves) * 100 if curves else None


def draw_panel(ax, suite, gc_key, steps, xlabel, ylim):
    handles, labels = [], []

    def _plot(curves, color, ls="-", lw=1.8, label=None):
        if curves is None or len(curves) == 0:
            return None
        mean = curves.mean(axis=0)
        std  = curves.std(axis=0)
        line, = ax.plot(steps, mean, color=color,
                        linestyle=ls, linewidth=lw, zorder=2)
        ax.fill_between(steps, mean - std, mean + std,
                        color=color, alpha=0.15, linewidth=0, zorder=1)
        if label:
            handles.append(line)
            labels.append(label)
        return line

    _plot(get_curves(suite, "hard_cbm", "lam_c1"), C_HARD, ls="-",  label="Hard CBM")
    _plot(get_curves(suite, "seq_cbm",  "lam_c1"), C_SEQ,  ls="--", label="Seq CBM")

    for lam, color, llab in zip(LAMS, C_JOINT, LAM_LABEL):
        _plot(get_curves(suite, "soft_cbm", lam), color, label=f"Joint CBM {llab}")

    for lam, color, llab in zip(LAMS, C_GCCBM, LAM_LABEL):
        _plot(get_curves(suite, gc_key, lam), color, label=f"GC-CBM {llab}")

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=9)
    return handles, labels


# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

tt_d  = joblib.load(master + "results/results_tabulartoy_cbm_suite.dict")
ds_d  = joblib.load(master + "results/results_dsprites_cbm_suite.dict")
cub_d = joblib.load(master + "results/results_cub_cbm_suite.dict")

datasets = [
    (axes[0], tt_d,  "acbm",   np.array([0,1,2,3]),  "Concepts intervened",       (45,  101), "TabularToy"),
    (axes[1], ds_d,  "gc_cbm", np.arange(6),          "Concepts intervened",       (93,  101), "dSprites"),
    (axes[2], cub_d, "gc_cbm", np.arange(29),         "Concept groups intervened", (10,  103), "CUB-200"),
]

legend_handles, legend_labels = None, None

for ax, suite, gc_key, steps, xlabel, ylim, title in datasets:
    h, l = draw_panel(ax, suite, gc_key, steps, xlabel, ylim)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if legend_handles is None:
        legend_handles, legend_labels = h, l

axes[0].set_ylabel("Task accuracy (%)", fontsize=10)

plt.tight_layout(rect=[0, 0.17, 1, 1])

fig.legend(
    legend_handles, legend_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),
    ncol=4,
    fontsize=9,
    framealpha=0.9,
    handlelength=2.0,
)

for ext in [".pdf", ".png"]:
    out = master + f"experiments/evaluate_models/intervention_cbm_combined{ext}"
    plt.savefig(out, bbox_inches="tight", dpi=150 if ext == ".png" else None)
    print(f"Saved → {out}")
plt.show()
