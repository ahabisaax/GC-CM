"""
Combined CEM intervention plot — TabularToy | dSprites | CUB-200 side by side.
Shared legend below the subplots.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import joblib

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master + "data/CUB200")

LAMS      = ["lam_c0.1", "lam_c0.5", "lam_c1"]
LAM_LABEL = [r"$\lambda_c=0.1$", r"$\lambda_c=0.5$", r"$\lambda_c=1.0$"]
C_CEM   = ["#f4a896", "#e05c3a", "#8b1a00"]
C_GCCEM = ["#92c4de", "#2b7ab8", "#003f7d"]

IV_KEY = "test_acc_y_random_group_level_True_use_prior_False_ints"

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_tt():
    d = joblib.load(master + "results/results_tabulartoy_acem_suite.dict")
    steps = np.array([0, 1, 2, 3])
    curves = {}
    for lam in LAMS:
        cem_curves, gc_curves = [], []
        for r in d.get("cem", {}).get(lam, {}).values():
            iv = r.get("interv", {}).get("random", [[]])[0]
            if iv:
                curve = np.array(iv) * 100
                cem_curves.append(curve)
        for r in d.get("acem", {}).get(lam, {}).values():
            iv = r.get("interv", {}).get("random", [[]])[0]
            if iv:
                curve = np.array(iv) * 100
                gc_curves.append(curve)
        curves[lam] = (np.array(cem_curves) if cem_curves else None,
                       np.array(gc_curves)  if gc_curves  else None)
    return steps, curves


def load_dsprites():
    d = joblib.load(master + "results/results_dsprites_suite.dict")
    steps = np.arange(6)
    curves = {}
    for lam in LAMS:
        cem_curves, gc_curves = [], []
        for r in d.get("cem", {}).get(lam, {}).values():
            iv = r.get("interv", {}).get("random", [[]])[0]
            if iv:
                curve = np.array(iv) * 100
                if "task_acc" in r:
                    curve[0] = r["task_acc"] * 100
                cem_curves.append(curve)
        for r in d.get("crcem", {}).get(lam, {}).values():
            iv = r.get("interv", {}).get("random", [[]])[0]
            if iv:
                curve = np.array(iv) * 100
                if "task_acc" in r:
                    curve[0] = r["task_acc"] * 100
                gc_curves.append(curve)
        curves[lam] = (np.array(cem_curves) if cem_curves else None,
                       np.array(gc_curves)  if gc_curves  else None)
    return steps, curves


def load_cub():
    folder = master + "results/cub_cem/"
    model_names = {
        "lam_c0.1": ("CEM_adam_lr1e-03_bs256_lam_c0.1",
                     "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c0.1_shared_critic"),
        "lam_c0.5": ("CEM_adam_lr1e-03_bs256_lam_c0.5",
                     "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c0.5_shared_critic"),
        "lam_c1":   ("CEM_adam_lr1e-03_bs256_lam_c1",
                     "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c1_shared_critic"),
    }
    n_groups = 28
    steps = np.arange(n_groups + 1)
    curves = {}
    for lam, (cem_name, gc_name) in model_names.items():
        cem_curves, gc_curves = [], []
        for mname, lst in [(cem_name, cem_curves), (gc_name, gc_curves)]:
            for fold in range(5):
                p = os.path.join(folder, mname,
                                 f"{mname}_split_{fold}_results.joblib")
                if os.path.exists(p):
                    sr = joblib.load(p)
                    iv = sr.get(IV_KEY)
                    if iv:
                        curve = np.array(iv) * 100
                        if "test_acc_y" in sr:
                            curve[0] = sr["test_acc_y"] * 100
                        lst.append(curve)
        curves[lam] = (np.array(cem_curves) if cem_curves else None,
                       np.array(gc_curves)  if gc_curves  else None)
    return steps, curves


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

datasets = [
    ("TabularToy",  *load_tt(),       "Concepts intervened",        (99.0, 100.2)),
    ("dSprites",    *load_dsprites(), "Concepts intervened",        (74,   100)),
    ("CUB-200",     *load_cub(),      "Concept groups intervened",  (65,   92)),
]

legend_handles = []
legend_labels  = []

for ax, (title, steps, curves, xlabel, ylim) in zip(axes, datasets):
    for lam, color, llab in zip(LAMS, C_CEM, LAM_LABEL):
        cem_c, _ = curves[lam]
        if cem_c is not None and len(cem_c):
            mean = cem_c.mean(axis=0)
            line, = ax.plot(steps, mean, color=color, linewidth=1.8, zorder=2)
            if ax is axes[0]:
                legend_handles.append(line)
                legend_labels.append(f"CEM {llab}")

    for lam, color, llab in zip(LAMS, C_GCCEM, LAM_LABEL):
        _, gc_c = curves[lam]
        if gc_c is not None and len(gc_c):
            mean = gc_c.mean(axis=0)
            line, = ax.plot(steps, mean, color=color, linewidth=1.8, zorder=2)
            if ax is axes[0]:
                legend_handles.append(line)
                legend_labels.append(f"GC-CEM {llab}")

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylim(*ylim)
    if ax is axes[0]:
        ax.set_xticks([0, 1, 2, 3])
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=9)

axes[0].set_ylabel("Task accuracy (%)", fontsize=10)

plt.tight_layout(rect=[0, 0.13, 1, 1])

fig.legend(
    legend_handles, legend_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),
    ncol=6,
    fontsize=9,
    framealpha=0.9,
    handlelength=2.0,
)

for ext in [".pdf", ".png"]:
    out = master + f"experiments/evaluate_models/intervention_cem_combined{ext}"
    plt.savefig(out, bbox_inches="tight", dpi=150 if ext == ".png" else None)
    print(f"Saved → {out}")
plt.show()
