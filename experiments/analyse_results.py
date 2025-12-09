import os
import glob
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np
import re


def parse_run_name_attributes(run_name):
    """
    Reverse-engineers your auto-generated run name to extract hyperparameters.
    Format example: CRCBM_adamw_lr0.005_bs4096_lam1.0_lagrangian_del0
    """
    attrs = {
        "Architecture": "Unknown",
        "Optimizer": "Unknown",
        "LR": np.nan,
        "Batch Size": "Unknown",
        "Lambda_adv": "0.0",
        "Scheduler": "None",
        "Delay": 0,
        "TTUR": False,
        "Lambda_C": '0.0'
    }

    # 1. Architecture
    if "HardCBM" in run_name:
        attrs["Architecture"] = "Independent"
    elif "SoftCBM" in run_name:
        attrs["Architecture"] = "Soft CBM"
    elif "CRCBM" in run_name:
        attrs["Architecture"] = "Adversarial CBM"

    # 2. Optimizer
    if "adamw" in run_name:
        attrs["Optimizer"] = "AdamW"
    elif "adam" in run_name:
        attrs["Optimizer"] = "Adam"
    elif "sgd" in run_name:
        attrs["Optimizer"] = "SGD"

    # 3. Batch Size (look for bsXXXX)
    bs_match = re.search(r"bs(\d+)", run_name)
    if bs_match: attrs["Batch Size"] = int(bs_match.group(1))

    lam_match = re.search(r"lam([\d\.]+)", run_name)
    if lam_match:
        try:
            attrs["Lambda_adv"] = float(lam_match.group(1))
        except:
            pass

    # 5. Lambda Concept (look for cwX.X or lam_cX.X depending on naming)
    # Based on generate_auto_run_name: "cw{c_weight}"
    cw_match = re.search(r"lam_c([\d\.]+)", run_name)
    if cw_match:
        try:
            attrs["Lambda_C"] = float(cw_match.group(1))
        except:
            pass

    # 5. Scheduler
    if "lagrange" in run_name:
        attrs["Scheduler"] = "Lagrangian"
    elif "constant" in run_name:
        attrs["Scheduler"] = "Constant"

    # 6. TTUR
    if "TTUR" in run_name: attrs["TTUR"] = True

    # 7. Learning Rate (This is tricky if TTUR is on, grabbing the first one)
    lr_match = re.search(r"lr([\d\.e-]+)", run_name)
    if lr_match:
        try:
            attrs["LR"] = float(lr_match.group(1))
        except:
            pass

    return attrs


def gather_results(results_dir):
    """
    Crawls directory and returns a Pandas DataFrame of all results.
    """
    print(f"Scanning {results_dir}...")
    # Matches the pattern used by run_experiments.py to save individual run results
    files = glob.glob(os.path.join(results_dir, "**", "*_results.joblib"), recursive=True)

    # Also look for the master results file if individual ones aren't found
    if not files:
        files = glob.glob(os.path.join(results_dir, "results.joblib"))

    data = []

    for f in files:
        try:
            split_match = re.search(r"split_(\d+)", f)
            if split_match:
                split_id = int(split_match.group(1))
            else:
                # If we can't find it, assume split 0 or try to infer from structure
                print(f"Warning: Could not infer split ID from {f}. Assuming split 0.")
                split_id = 0

            metrics = joblib.load(f)



            row = {
                "Run Name": f.split("/")[-1],
                "Split": split_id,
                "Path": f
            }

            # --- Extract Metrics (Add more here as needed) ---
            # Use .get() to handle missing metrics safely
            row["Task Acc"] = metrics.get("test_acc_y", np.nan)
            row["Concept Acc"] = metrics.get("test_acc_c", np.nan)
            row["Task AUC"] = metrics.get("test_auc_y", np.nan)
            row["Concept AUC"] = metrics.get("test_auc_c", np.nan)

            # Leakage Metrics (Prioritize Nats if available)
            row["CTL_normalised"] = metrics.get("test_normalised_ctl",np.nan)
            row["ICL_normalised"] = metrics.get("test_normalised_icl",np.nan)
            row['ICL_unnormalisaed'] = metrics.get('test_icl_average')
            # Decomposed ICL (If available)
            row["ICL (Task)"] = metrics.get("test_icl_task", np.nan)
            row["ICL (Input)"] = metrics.get("test_icl_cmi", np.nan)

            # --- Extract Hyperparameters ---
            # Parse the string name into columns
            filename = f.split("/")[-1]
            row.update(parse_run_name_attributes(filename.split('.joblib')[0]))

            data.append(row)

        except Exception as e:
            print(f"Skipping {f}: {e}")

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} runs.")
    return df


def plot_ablation(df, x_axis, y_axis, hue="Architecture", filename="plot.png", title=None):
    """
    Creates a bar chart with Error Bars (Confidence Intervals).
    """
    if df.empty:
        print(f"No data to plot for {filename}")
        return

    plt.figure(figsize=(10, 6))

    # Filter out NaNs for the plotting columns
    plot_df = df.dropna(subset=[x_axis, y_axis])

    if plot_df.empty:
        print(f"Data empty after filtering NaNs for {x_axis}/{y_axis}")
        return

    # Seaborn Barplot
    # errorbar='ci' gives 95% Confidence Interval
    # errorbar='sd' gives Standard Deviation (often better for ML to show spread)
    sns.barplot(
        data=plot_df,
        x=x_axis,
        y=y_axis,
        hue=hue,
        errorbar="sd",
        capsize=0.1,
        palette="viridis",
        edgecolor="black"
    )

    if title:
        plt.title(title)
    else:
        plt.title(f"{y_axis} vs {x_axis}")

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Made results_dir optional for easier IDE debugging
    parser.add_argument("--results_dir", required=False, default=None, help="Path to results directory")
    parser.add_argument("--output_dir", default="analysis_plots")
    args = parser.parse_args()

    # DEBUG CONFIGURATION
    # If no arg provided, use this default path (Change this to your local path!)
    DEBUG_DEFAULT_PATH = "results/tabulartoy_25_10k_models/auto"

    if args.results_dir:
        results_dir = args.results_dir
    else:
        print(f"No --results_dir provided. Using debug default: {DEBUG_DEFAULT_PATH}")
        results_dir = DEBUG_DEFAULT_PATH

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    if not os.path.exists(results_dir):
        print(f"ERROR: Results directory '{results_dir}' does not exist.")
        print("Please set --results_dir or update DEBUG_DEFAULT_PATH in the script.")
    else:
        df = gather_results(results_dir)

        if not df.empty:
            # Save raw CSV for manual checking
            csv_path = os.path.join(args.output_dir, "aggregated_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved raw data to {csv_path}")

            # --- 2. Generate Standard Plots ---

            # A. Batch Size Ablation (Filtered for CRCBM)
            # Goal: Show that higher batch size -> Lower CTL
            if "Architecture" in df.columns:
                crcbm_df = df[df["Architecture"] == "Adversarial CBM"]
                if not crcbm_df.empty:
                    plot_ablation(
                        crcbm_df,
                        x_axis="Batch Size",
                        y_axis="CTL_normalised",
                        title="Effect of Batch Size on Leakage (CRCBM)",
                        filename=os.path.join(args.output_dir, "batch_size_vs_ctl.png")
                    )

            # B. Model Comparison (The "Money Plot")
            # Goal: Compare Soft vs Hard vs CRCBM on Accuracy and CTL
            # We group by Architecture to show the overall performance classes
            plot_ablation(
                df,
                x_axis="Architecture",
                y_axis="Task Acc",
                hue=None,  # Just one bar per arch
                title="Task Accuracy by Model Type",
                filename=os.path.join(args.output_dir, "comparison_task_acc.png")
            )

            plot_ablation(
                df,
                x_axis="Architecture",
                y_axis="CTL_normalised",
                hue=None,
                title="Leakage (CTL) by Model Type",
                filename=os.path.join(args.output_dir, "comparison_ctl.png")
            )

            # C. Scheduler Comparison
            # Goal: Constant vs Lagrangian
            if "Architecture" in df.columns:
                crcbm_df = df[df["Architecture"] == "Adversarial CBM"]
                if not crcbm_df.empty:
                    plot_ablation(
                        crcbm_df,
                        x_axis="Scheduler",
                        y_axis="Task Acc",
                        title="Scheduler Stability Comparison",
                        filename=os.path.join(args.output_dir, "scheduler_acc.png")
                    )

            print("Analysis Complete.")
        else:
            print("DataFrame is empty. No plots generated.")