"""
Task accuracy and concept accuracy across datasets, with lambda variation.

CBM, ACBM and GC-CBM are shown as grouped bars across λ_c ∈ {0.1, 0.5, 1.0}.
  ACBM   = adversarial CBM with gradient penalty, no shared critic
  GC-CBM = adversarial CBM with GRL + shared critic
HardCBM and SeqCBM (fixed λ_c = 1) are shown as horizontal reference lines.

Layout: 2 rows (task / concept) × 3 columns (TabularToy, dSprites, CUB-200).

Run from project root:
  python experiments/evaluate_models/plot_task_accuracy.py
"""
import os, sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

master_folder = os.getcwd().replace("/experiments/evaluate_models", "") + "/"

# ---------------------------------------------------------------------------
# Wong colorblind-safe palette
# ---------------------------------------------------------------------------
C_CBM    = "#0072B2"   # blue
C_ACBM   = "#CC79A7"   # mauve/pink
C_GCCBM  = "#D55E00"   # vermillion
C_HARD   = "#009E73"   # green
C_SEQ    = "#E69F00"   # amber

LAM_C_VALS   = [0.1, 0.5, 1.0]
LAM_C_LABELS = [r"$\lambda_c = 0.1$", r"$\lambda_c = 0.5$", r"$\lambda_c = 1.0$"]
DATASET_ORDER = ["TabularToy", "dSprites", "CUB-200"]

# ---------------------------------------------------------------------------
# Source paths — helpers
# ---------------------------------------------------------------------------
def _p(folder, mdir):
    return master_folder + folder, mdir

SOURCES = {
    "TabularToy": {
        "cbm": {
            0.1: _p("results/tabulartoy_25_10k_models", "SoftCBM_adam_lr0.05_bs64_lam_c0.1"),
            0.5: _p("results/tabulartoy_25_10k_models", "SoftCBM_adam_lr0.05_bs64_lam_c0.5"),
            1.0: _p("results/tabulartoy_25_10k_models", "SoftCBM_adam_lr0.05_bs64_lam_c1"),
        },
        "acbm": {
            0.1: _p("results/tabulartoy_25_10k_models", "CRCBM_adam_lr0.05_bs64_lam1_none_lam_c0.1"),
            0.5: _p("results/tabulartoy_25_10k_models", "CRCBM_adam_lr0.05_bs64_lam1_none_lam_c0.5"),
            1.0: _p("results/tabulartoy_25_10k_models", "CRCBM_adam_lr0.05_bs64_lam1_none_lam_c1"),
        },
        "gccbm": {
            0.1: _p("results/tabulartoy_25_10k_models_acbm_shared_critic", "CRCBM_adam_lr0.01_bs64_lam1_none_lam_c0.1_shared_critic"),
            0.5: _p("results/tabulartoy_25_10k_models_acbm_shared_critic", "CRCBM_adam_lr0.01_bs64_lam1_none_lam_c0.5_shared_critic"),
            1.0: _p("results/tabulartoy_25_10k_models_acbm_shared_critic", "CRCBM_adam_lr0.01_bs64_lam1_none_lam_c1_shared_critic"),
        },
        "hard": _p("results/tabulartoy_25_10k_models", "HardCBM_adam_lr0.05_bs64_lam_c1"),
        "seq":  _p("results/tabulartoy_25_10k_models_acbm_shared_critic", "SeqCBM_adam_lr0.05_bs64_lam_c1"),
    },
    "dSprites": {
        "cbm": {
            0.1: _p("results/dsprites_dep_0_models", "SoftCBM_adam_lr1e-04_bs64_lam_c0.1"),
            0.5: _p("results/dsprites_dep_0_models", "SoftCBM_adam_lr1e-04_bs64_lam_c0.5"),
            1.0: _p("results/dsprites_dep_0_models", "SoftCBM_adam_lr1e-04_bs64_lam_c1"),
        },
        "acbm": {
            0.1: _p("results/dsprites_dep_0_models", "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c0.1"),
            0.5: _p("results/dsprites_dep_0_models", "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c0.5"),
            1.0: _p("results/dsprites_dep_0_models", "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c1"),
        },
        "gccbm": {
            0.1: _p("results/dsprites_ACBM_shared_critic", "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c0.1_shared_critic"),
            0.5: _p("results/dsprites_ACBM_shared_critic", "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c0.5_shared_critic"),
            1.0: _p("results/dsprites_ACBM_shared_critic", "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c1_shared_critic"),
        },
        "hard": _p("results/dsprites_dep_0_models", "HardCBM_adam_lr1e-04_bs64_lam_c1"),
        "seq":  _p("results/dsprites_sequential_acbm", "SeqCBM_adam_lr1e-04_bs64_lam_c1"),
    },
    "CUB-200": {
        "cbm": {
            0.1: _p("results/cub_soft_0.01", "SoftCBM_adam_lr1e-04_bs256_lam_c0.1"),
            0.5: _p("results/cub_soft_0.01", "SoftCBM_adam_lr1e-04_bs256_lam_c0.5"),
            1.0: _p("results/cub_soft_0.01", "SoftCBM_adam_lr1e-04_bs256_lam_c1"),
        },
        "acbm": {
            0.1: _p("results/cub_soft_0.01", "CRCBM_adam_lr1e-04_bs256_lam1.0_none_lam_c0.1"),
            0.5: _p("results/cub_soft_0.01", "CRCBM_adam_lr1e-04_bs256_lam1.0_none_lam_c0.5"),
            1.0: _p("results/cub_soft_0.01", "CRCBM_adam_lr1e-04_bs256_lam1.0_none_lam_c1"),
        },
        "gccbm": {
            0.1: _p("results/cub_acbm_shared_critic", "ACBM_adam_lr1e-04_bs256_lam1_none_lam_c0.1_shared_critic"),
            0.5: _p("results/cub_acbm_shared_critic", "ACBM_adam_lr1e-04_bs256_lam1_none_lam_c0.5_shared_critic"),
            1.0: _p("results/cub_acbm_shared_critic", "ACBM_adam_lr1e-04_bs256_lam1_none_lam_c1_shared_critic"),
        },
        "hard": _p("results/cub_soft_0.01", "HardCBM_adam_lr1e-04_bs256_lam_c1"),
        "seq":  _p("results/cub_acbm_shared_critic", "SeqCBM_adam_lr1e-04_bs256_lam_c1"),
    },
}


def load_metrics(folder, mdir):
    path = os.path.join(folder, mdir)
    splits = sorted(glob.glob(os.path.join(path, "*_split_*_results.joblib")))
    task, concept = [], []
    for s in splits:
        d = joblib.load(s)
        task.append(d.get("test_acc_y", np.nan))
        concept.append(d.get("test_acc_c", np.nan))
    return np.array(task), np.array(concept)


def mean_sem(vals):
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return np.mean(vals), np.std(vals) / np.sqrt(max(len(vals) - 1, 1))


# ---------------------------------------------------------------------------
# Pre-load all data
# ---------------------------------------------------------------------------
bar_models = ["cbm", "acbm", "gccbm"]
bar_data   = {ds: {m: {} for m in bar_models} for ds in DATASET_ORDER}
ref_data   = {ds: {} for ds in DATASET_ORDER}

for ds in DATASET_ORDER:
    for model in bar_models:
        for lam in LAM_C_VALS:
            folder, mdir = SOURCES[ds][model][lam]
            t, c = load_metrics(folder, mdir)
            bar_data[ds][model][lam] = (mean_sem(t), mean_sem(c))
    for model in ("hard", "seq"):
        folder, mdir = SOURCES[ds][model]
        t, c = load_metrics(folder, mdir)
        ref_data[ds][model] = (mean_sem(t), mean_sem(c))

# ---------------------------------------------------------------------------
# Shared plot settings
# ---------------------------------------------------------------------------
n_bars  = len(bar_models)  # 3
bar_w   = 0.22
offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_w
xs      = np.arange(len(LAM_C_VALS))

BAR_COLORS = {"cbm": C_CBM, "acbm": C_ACBM, "gccbm": C_GCCBM}
BAR_LABELS = {"cbm": "CBM", "acbm": "ACBM", "gccbm": "GC-CBM"}

YLIMS = {
    0: {"TabularToy": (0.96, 1.005), "dSprites": (0.88, 1.005), "CUB-200": (0.55, 0.80)},
    1: {"TabularToy": (0.40, 1.02),  "dSprites": (0.93, 1.005), "CUB-200": (0.60, 1.01)},
}

PLOT_DIR = master_folder + "results/plots/cbm/"
os.makedirs(PLOT_DIR, exist_ok=True)


def make_figure(metric_idx, ylabel, title, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False, constrained_layout=True)

    for col, ds in enumerate(DATASET_ORDER):
        ax = axes[col]

        for i, model in enumerate(bar_models):
            means = [bar_data[ds][model][l][metric_idx][0] for l in LAM_C_VALS]
            sems  = [bar_data[ds][model][l][metric_idx][1] for l in LAM_C_VALS]
            ax.bar(
                xs + offsets[i], means, width=bar_w,
                color=BAR_COLORS[model], label=BAR_LABELS[model],
                alpha=0.88, edgecolor="white", linewidth=0.5,
            )
            ax.errorbar(
                xs + offsets[i], means, yerr=sems,
                fmt="none", color="black", capsize=3, linewidth=0.9,
            )

        x_lo, x_hi = xs[0] - bar_w * 1.8, xs[-1] + bar_w * 1.8
        for ref_model, color, ls, lw, ref_label, va in [
            ("hard", C_HARD, "--", 1.6, "HardCBM", "bottom"),
            ("seq",  C_SEQ,  ":",  1.8, "SeqCBM",  "top"),
        ]:
            val = ref_data[ds][ref_model][metric_idx][0]
            sem = ref_data[ds][ref_model][metric_idx][1]
            ax.hlines(val, x_lo, x_hi, colors=color, linestyles=ls,
                      linewidth=lw, label=ref_label, zorder=5)
            ax.fill_between([x_lo, x_hi], [val - sem] * 2, [val + sem] * 2,
                            color=color, alpha=0.12, zorder=4)
            ax.text(x_hi + 0.04, val, f"{val:.3f}",
                    color=color, fontsize=7.5, va=va, ha="left",
                    fontweight="bold", zorder=6)

        ax.set_xticks(xs)
        ax.set_xticklabels(LAM_C_LABELS, fontsize=10)
        ax.set_xlim(x_lo - 0.05, x_hi + 0.05)

        if ds in YLIMS[metric_idx]:
            ax.set_ylim(*YLIMS[metric_idx][ds])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
        ax.set_title(ds, fontsize=12, fontweight="bold", pad=6)

        if col == 0:
            ax.set_ylabel(ylabel, fontsize=11)
            ax.legend(fontsize=9, loc="lower left", framealpha=0.9,
                      frameon=True, edgecolor="0.8", ncol=3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generate the two separate figures
# ---------------------------------------------------------------------------
make_figure(
    metric_idx=0,
    ylabel="Task accuracy",
    title="Task accuracy across datasets and " + r"$\lambda_c$",
    save_path=PLOT_DIR + "task_accuracy_lambda.png",
)

make_figure(
    metric_idx=1,
    ylabel="Concept accuracy",
    title="Concept accuracy across datasets and " + r"$\lambda_c$",
    save_path=PLOT_DIR + "concept_accuracy_lambda.png",
)

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
for metric_idx, label in [(0, "Task accuracy"), (1, "Concept accuracy")]:
    print(f"\n=== {label} ===")
    print(f"{'Model':16s}", "  ".join(f"{ds:>26s}" for ds in DATASET_ORDER))
    for lam in LAM_C_VALS:
        for model_key, model_name in [("cbm", "CBM"), ("acbm", "ACBM"), ("gccbm", "GC-CBM")]:
            row_label = f"{model_name} λ={lam}"
            row = f"{row_label:16s}"
            for ds in DATASET_ORDER:
                m, s = bar_data[ds][model_key][lam][metric_idx]
                row += f"  {m:.4f}±{s:.4f}              "
            print(row)
    for model_key, model_name in [("hard", "HardCBM"), ("seq", "SeqCBM")]:
        row = f"{model_name:16s}"
        for ds in DATASET_ORDER:
            m, s = ref_data[ds][model_key][metric_idx]
            row += f"  {m:.4f}±{s:.4f}              "
        print(row)
