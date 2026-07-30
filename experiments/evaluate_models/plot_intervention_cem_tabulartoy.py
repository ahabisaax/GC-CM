"""CEM intervention plot — TabularToy."""
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
d = joblib.load(master + "results/results_tabulartoy_acem_suite.dict")

STEPS     = np.array([0, 1, 2, 3])
LAMS      = ["lam_c0.1", "lam_c0.5", "lam_c1"]
LAM_LABEL = [r"$\lambda_c=0.1$", r"$\lambda_c=0.5$", r"$\lambda_c=1.0$"]
C_CEM   = ["#f4a896", "#e05c3a", "#8b1a00"]
C_GCCEM = ["#92c4de", "#2b7ab8", "#003f7d"]


def get_curves(suite, model_key, lam):
    curves = []
    for r in suite.get(model_key, {}).get(lam, {}).values():
        iv = r.get("interv", {}).get("random", [[]])[0]
        if iv:
            curves.append(np.array(iv) * 100)
    return np.array(curves) if curves else None


def plot_curve(ax, curves, color, linestyle="-", lw=2.0, label=None):
    if curves is None or len(curves) == 0:
        return
    mean = curves.mean(axis=0)
    std  = curves.std(axis=0)
    ax.plot(STEPS, mean, color=color, linestyle=linestyle,
            linewidth=lw, label=label, zorder=2)
    ax.fill_between(STEPS, mean - std, mean + std,
                    color=color, alpha=0.15, linewidth=0, zorder=1)


fig, ax = plt.subplots(figsize=(6, 4))

for lam, color, llab in zip(LAMS, C_CEM, LAM_LABEL):
    plot_curve(ax, get_curves(d, "cem", lam), color, label=f"CEM {llab}")

for lam, color, llab in zip(LAMS, C_GCCEM, LAM_LABEL):
    plot_curve(ax, get_curves(d, "acem", lam), color, label=f"GC-CEM {llab}")

ax.set_xlabel("Concepts intervened", fontsize=12)
ax.set_ylabel("Task accuracy (%)", fontsize=12)
ax.set_xticks(STEPS)
ax.set_xlim(-0.1, 3.1)
ax.set_ylim(99, 100.2)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.legend(fontsize=8.5, loc="upper left", ncol=2, framealpha=0.9)

plt.tight_layout()
for ext in [".pdf", ".png"]:
    out = master + f"experiments/evaluate_models/intervention_cem_tabulartoy{ext}"
    plt.savefig(out, bbox_inches="tight", dpi=150 if ext == ".png" else None)
    print(f"Saved → {out}")
plt.show()
