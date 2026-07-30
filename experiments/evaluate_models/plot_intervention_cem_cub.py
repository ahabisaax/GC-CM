"""CEM intervention plot — CUB-200."""
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"

IV_KEY = "test_acc_y_random_group_level_True_use_prior_False_ints"
CUB_CEM_FOLDER = master + "results/cub_cem/"

LAMS      = ["lam_c0.1", "lam_c0.5", "lam_c1"]
LAM_LABEL = [r"$\lambda_c=0.1$", r"$\lambda_c=0.5$", r"$\lambda_c=1.0$"]
C_CEM   = ["#f4a896", "#e05c3a", "#8b1a00"]
C_GCCEM = ["#92c4de", "#2b7ab8", "#003f7d"]

MODEL_NAMES = {
    "lam_c0.1": ("CEM_adam_lr1e-03_bs256_lam_c0.1",
                 "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c0.1_shared_critic"),
    "lam_c0.5": ("CEM_adam_lr1e-03_bs256_lam_c0.5",
                 "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c0.5_shared_critic"),
    "lam_c1":   ("CEM_adam_lr1e-03_bs256_lam_c1",
                 "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c1_shared_critic"),
}

n_groups = 28
STEPS = np.arange(n_groups + 1)


def get_curves_joblib(model_name):
    curves = []
    for fold in range(5):
        p = os.path.join(CUB_CEM_FOLDER, model_name,
                         f"{model_name}_split_{fold}_results.joblib")
        if os.path.exists(p):
            sr = joblib.load(p)
            iv = sr.get(IV_KEY)
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
    cem_name, _ = MODEL_NAMES[lam]
    plot_curve(ax, get_curves_joblib(cem_name), color, label=f"CEM {llab}")

for lam, color, llab in zip(LAMS, C_GCCEM, LAM_LABEL):
    _, gc_name = MODEL_NAMES[lam]
    plot_curve(ax, get_curves_joblib(gc_name), color, label=f"GC-CEM {llab}")

ax.set_xlabel("Concept groups intervened", fontsize=12)
ax.set_ylabel("Task accuracy (%)", fontsize=12)
ax.set_xlim(-0.3, n_groups + 0.3)
ax.set_ylim(65, 92)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.legend(fontsize=8.5, loc="lower right", ncol=2, framealpha=0.9)

plt.tight_layout()
for ext in [".pdf", ".png"]:
    out = master + f"experiments/evaluate_models/intervention_cem_cub{ext}"
    plt.savefig(out, bbox_inches="tight", dpi=150 if ext == ".png" else None)
    print(f"Saved → {out}")
plt.show()
