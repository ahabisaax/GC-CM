"""
Plot mean training time (hours) per model type for CUB and TabularToy datasets.
Training time is read from *_training_times.npy files saved during training.
Each file contains [total_seconds, num_epochs].
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config: results directories and model-type labels
# ---------------------------------------------------------------------------

RESULTS = {
    "CUB": [
        "results/cub_soft_0.01",
        "results/cub_acbm_shared_critic",
    ],
    "TabularToy": [
        "results/tabulartoy_25_10k_models",
        "results/tabulartoy_25_10k_models_acbm_shared_critic",
        "results/tabulartoy_25_10k_models_sequential_cbm",
    ],
}

# Ordered model labels for the plot
MODEL_ORDER = [
    "Joint CBM",
    "Hard CBM",
    "Sequential CBM",
    "ACBM (shared critic)",
    "CRCBM",
]

SKIP = {"auto", "Untitled Folder"}
SKIP_EXTS = {".png", ".dict", ".joblib", ".txt"}


def classify_model(name: str) -> str | None:
    """Map a result directory name to a model label, or None to skip."""
    if name.startswith("SoftCBM"):
        return "Joint CBM"
    if name.startswith("HardCBM"):
        return "Hard CBM"
    if name.startswith("SeqCBM"):
        return "Sequential CBM"
    if name.startswith("ACBM") and "shared_critic" in name:
        return "ACBM (shared critic)"
    if name.startswith("CRCBM") and "shared_critic" not in name:
        return "CRCBM"
    return None


# ---------------------------------------------------------------------------
# Collect timing data
# ---------------------------------------------------------------------------

def collect_times(result_dirs):
    """Return dict: model_label -> list of per-fold seconds (across all variants)."""
    model_times: dict[str, list[float]] = {}
    for d in result_dirs:
        if not os.path.isdir(d):
            continue
        for model_dir in sorted(os.listdir(d)):
            if model_dir in SKIP or any(model_dir.endswith(e) for e in SKIP_EXTS):
                continue
            label = classify_model(model_dir)
            if label is None:
                continue
            time_files = glob.glob(f"{d}/{model_dir}/*_training_times.npy")
            fold_times = [np.load(f)[0] for f in time_files]
            if not fold_times:
                continue
            model_times.setdefault(label, []).extend(fold_times)
    return model_times


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_training_times(save_path="results/training_times_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, (dataset, result_dirs) in zip(axes, RESULTS.items()):
        model_times = collect_times(result_dirs)

        labels, means, stds = [], [], []
        for label in MODEL_ORDER:
            if label not in model_times:
                continue
            times_h = np.array(model_times[label]) / 3600
            labels.append(label)
            means.append(np.mean(times_h))
            stds.append(np.std(times_h))

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=stds, capsize=4, width=0.55,
                      color=plt.cm.tab10(np.linspace(0, 0.6, len(labels))),
                      edgecolor="black", linewidth=0.7, error_kw={"linewidth": 1.2})

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
        ax.set_ylabel("Training time (hours)", fontsize=11)
        ax.set_title(dataset, fontsize=13, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)

        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + std + 0.02 * ax.get_ylim()[1],
                    f"{mean:.2f}h", ha="center", va="bottom", fontsize=8.5)

    fig.suptitle("Mean training time per model (± std across folds & concept loss weights)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_training_times()
