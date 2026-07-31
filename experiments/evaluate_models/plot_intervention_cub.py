"""
Intervention plot for CUB-200 — CBM family only.

Single plot with:
  Hard CBM   — single line (no λ_c)
  Seq CBM    — single line (no λ_c)
  Joint CBM  — three lines, one per λ_c (red shades)
  GC-CBM     — three lines, one per λ_c (blue shades)

CUB has 28 concept groups; x-axis is 0–28 groups intervened.

Run from project root:
    python experiments/evaluate_models/plot_intervention_cub.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

master_folder = os.getcwd().replace("/experiments/evaluate_models", "") + "/"

cbm_d = joblib.load(master_folder + "results/results_cub_cbm_suite.dict")


def get_curves(suite, model_key, lam_key):
    curves = []
    for fold_r in suite.get(model_key, {}).get(lam_key, {}).values():
        iv = fold_r.get("interv", {}).get("random", [])
        if iv and len(iv[0]) > 0:
            curves.append(np.array(iv[0]))
    return np.array(curves) * 100 if curves else None


n_groups = 28
STEPS = np.arange(n_groups + 1)

LAMS      = ["lam_c0.1", "lam_c0.5", "lam_c1"]
LAM_LABEL = [r"$\lambda_c=0.1$", r"$\lambda_c=0.5$", r"$\lambda_c=1.0$"]

C_HARD  = "#2c2c2c"
C_SEQ   = "#7f7f7f"
C_JOINT = ["#f4a896", "#e05c3a", "#8b1a00"]
C_GCCBM = ["#92c4de", "#2b7ab8", "#003f7d"]


def plot_curve(ax, curves_arr, color, linestyle="-", lw=2.0, label=None, zorder=2):
    if curves_arr is None or len(curves_arr) == 0:
        return
    mean = curves_arr.mean(axis=0)
    std  = curves_arr.std(axis=0)
    ax.plot(STEPS, mean, color=color, linestyle=linestyle,
            linewidth=lw, label=label, zorder=zorder)
    ax.fill_between(STEPS, mean - std, mean + std,
                    color=color, alpha=0.15, linewidth=0, zorder=zorder - 1)


fig, ax = plt.subplots(figsize=(6, 4))

plot_curve(ax, get_curves(cbm_d, "hard_cbm", "lam_c1"),
           C_HARD, linestyle="-",  lw=2.0, label="Hard CBM")
plot_curve(ax, get_curves(cbm_d, "seq_cbm", "lam_c1"),
           C_SEQ,  linestyle="--", lw=2.0, label="Seq CBM")

for lam, color, llab in zip(LAMS, C_JOINT, LAM_LABEL):
    plot_curve(ax, get_curves(cbm_d, "soft_cbm", lam),
               color, lw=2.0, label=f"Joint CBM {llab}")

for lam, color, llab in zip(LAMS, C_GCCBM, LAM_LABEL):
    plot_curve(ax, get_curves(cbm_d, "gc_cbm", lam),
               color, lw=2.0, label=f"GC-CBM {llab}")

ax.set_xlabel("Concept groups intervened", fontsize=12)
ax.set_ylabel("Task accuracy (%)", fontsize=12)
ax.set_xlim(-0.3, n_groups + 0.3)
ax.set_ylim(10, 103)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.legend(fontsize=8.5, loc="lower left", ncol=2, framealpha=0.9)

plt.tight_layout()
out     = master_folder + "experiments/evaluate_models/intervention_cub.pdf"
out_png = out.replace(".pdf", ".png")
plt.savefig(out, bbox_inches="tight")
plt.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"Saved → {out}")
print(f"Saved → {out_png}")
plt.show()
