import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def learning_rate_check(df):
    def get_model(f):
        if 'CRCBM' in f: return 'CRCBM'
        if 'SoftCBM' in f: return 'SoftCBM'
        if 'Hard' in f: return 'HardCBM'
        return 'Other'
    
    # Work on a copy to avoid modifying the original dataframe
    df = df.copy()
    df['Model'] = df['filename'].apply(get_model)
    
    # Extract attributes
    df['LR'] = df['filename'].str.extract(r'lr([0-9\.eE-]+)_').astype(float)
    df['Lambda_C'] = df['filename'].str.extract(r'lam_c([0-9\.]+)_').astype(float)
    
    # Filter for Soft CBM
    df_plot = df[df["Model"] == "SoftCBM"].copy()
    
    metrics = [
        ('val_acc_y', 'Task Accuracy'), # Changed to val_acc_y based on previous context, check your column names!
        ('val_acc_c', 'Concept Accuracy'),
        ('val_ctl_average', 'Norm. CTL (Leakage)'),
        # ('test_normalised_icl', 'Norm. ICL (Inter-leakage)') # Add back if column exists
    ]
    
    for metric_col, title in metrics:
        # Check if column exists to avoid crash
        if metric_col not in df_plot.columns:
            print(f"Skipping {metric_col} - not in dataframe")
            continue

        plt.figure(figsize=(12, 7)) # Made slightly wider to fit labels
        
        ax = sns.barplot(
            data=df_plot,
            x="Lambda_C",
            y=metric_col,
            hue="LR",
            errorbar="sd",
            palette="viridis",
            edgecolor="black",
            capsize=0.1
        )
        
        # --- Add Labels ---
        # Iterate through each bar container (one for each hue category)
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)
        
        plt.title(f"Soft CBM: {title} vs Concept Weight")
        plt.xlabel(r"Concept Weight ($\lambda_c$)")
        plt.ylabel(title)
        plt.legend(title="Learning Rate", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

import os
import joblib
import pandas as pd
import re

def make_df(path):
    data_records = []
    
    # Define regex patterns to find the specific values in the filename
    # matches 'lam_c' followed by digits/dots
    lam_c_pattern = r"lam_c([0-9\.]+)"
    # matches 'split_' followed by digits
    split_pattern = r"split_([0-9]+)"  

    #if model == 'CRCBM':
    for f_dir in os.listdir(path):
        #if model in f_dir:
        model_path = os.path.join(path, f_dir)
            
        # Check if it's actually a directory
        if not os.path.isdir(model_path):
            continue

        for f in os.listdir(model_path):
            if 'results.joblib' in f:
                # 1. Extract Metadata using Regex
                lam_c_match = re.search(lam_c_pattern, f)
                split_match = re.search(split_pattern, f)
                
                # safely get values if matches are found
                lam_c_val = float(lam_c_match.group(1)) if lam_c_match else None
                split_val = int(split_match.group(1)) if split_match else None
                
                # 2. Load the actual result content
                result_content = joblib.load(os.path.join(model_path, f))
                
                # 3. Create a dictionary for this row
                row = {
                    'lam_c': lam_c_val,
                    'split': split_val,
                    'filename': f  # good to keep for debugging
                }
                
                # 4. Merge with result content
                # If result_content is a dict (e.g. {'acc': 0.5}), this adds those columns to the row
                if isinstance(result_content, dict):
                    row.update(result_content)
                else:
                    # If it's just a single number or object, store it in a generic column
                    row['result_obj'] = result_content
                
                data_records.append(row)

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(data_records)
    
    # Optional: Sort by lam_c and split for cleaner viewing
    if not df.empty:
        df = df.sort_values(by=['lam_c', 'split']).reset_index(drop=True)
        
    return df

# Example usage:
# df = make_df('CRCBM', '/path/to/your/results')
# print(df.head())



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_full_evaluation(df, dataset_name="Dataset"):
    def get_model(f):
        if 'CRCBM' in f: return 'CRCBM'
        if 'SoftCBM' in f: return 'SoftCBM'
        if 'Hard' in f: return 'HardCBM'
        return 'Other'
    
    df = df.copy()
    df['Model'] = df['filename'].apply(get_model)
    
    # Define metrics: A mix of performance and leakage
    metrics = [
        ('test_acc_y', 'Task Accuracy'),
        ('test_acc_c', 'Concept Accuracy'),
        ('test_normalised_ctl', 'Norm. CTL (Leakage)'),
        ('test_normalised_icl', 'Norm. ICL (Inter-leakage)')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Filter for only the two comparison models
    plot_df = df[df['Model'].isin(['CRCBM', 'SoftCBM'])]
    
    # Define style mappings to avoid KeyErrors
    model_styles = {
        "CRCBM": "o", 
        "SoftCBM": "s"
    }
    model_lines = {
        "CRCBM": "-", 
        "SoftCBM": "--"
    }

    for i, (col, title) in enumerate(metrics):
        ax = axes[i]
        
        # Pointplot for discrete lambda values
        sns.pointplot(
            data=plot_df,
            x='lam_c', 
            y=col, 
            hue='Model',
            markers=[model_styles[m] for m in plot_df['Model'].unique()],
            linestyles=[model_lines[m] for m in plot_df['Model'].unique()],
            capsize=.1, 
            errorbar='sd',
            ax=ax
        )
        
        # Calculate HardCBM Baseline (constant across X-axis)
        hard_subset = df[df['Model'] == 'HardCBM'][col]
        if not hard_subset.empty:
            hard_val = hard_subset.mean()
            hard_std = hard_subset.std()
            
            ax.axhline(hard_val, color='black', linestyle=':', label='HardCBM Baseline')
            # Shade the baseline error
            xlims = ax.get_xlim()
            ax.fill_between(xlims, hard_val - hard_std, hard_val + hard_std, 
                            color='black', alpha=0.1)
            ax.set_xlim(xlims) 

        ax.set_title(f"{dataset_name}: {title}", fontweight='bold')
        ax.set_ylabel("Value")
        ax.set_xlabel(r"$\lambda_c$")
        
        if i == 0:
            ax.legend(title="Model Type")
        else:
            ax.get_legend().remove()

    plt.tight_layout()
    plt.show()

# Run the fixed function
# Define this globally or inside your function
    custom_palette = {
        'CRCBM':   '#1f77b4',   # Blue
        'SoftCBM': '#ff7f0e',   # Orange
        'HardCBM': '#2ca02c',   # Green
        'CRCEM':   '#d62728',   # Red
        'CEM':     '#9467bd',   # Purple
        'Other':   '#7f7f7f'    # Grey
    }

def plot_bar_evaluation(df, dataset_name="Dataset"):
    def get_model(f):
        if 'CRCBM' in f: return 'CRCBM'
        if 'SoftCBM' in f: return 'SoftCBM'
        if 'Hard' in f: return 'HardCBM'
        return 'Other'
    
    df = df.copy()
    df['Model'] = df['filename'].apply(get_model)
    
    metrics = [
        ('test_acc_y', 'Task Accuracy'),
        ('test_acc_c', 'Concept Accuracy'),
        ('test_normalised_ctl', 'Norm. CTL (Leakage)'),
        ('test_normalised_icl', 'Norm. ICL (Inter-leakage)')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Filter for Soft and CRCBM
    plot_df = df[df['Model'].isin(['CRCBM', 'SoftCBM'])]
    
    for i, (col, title) in enumerate(metrics):
        ax = axes[i]
        
        # USE THE FIXED PALETTE HERE
        barplot = sns.barplot(
            data=plot_df,
            x='lam_c', 
            y=col, 
            hue='Model',
            hue_order=['CRCBM', 'SoftCBM'], # Fixes the order of the bars too
            palette=custom_palette, 
            capsize=.1, 
            errorbar='sd',
            ax=ax
        )
        
        for container in barplot.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
        
        # Baseline logic
        hard_subset = df[df['Model'] == 'HardCBM'][col]
        if not hard_subset.empty:
            hard_val = hard_subset.mean()
            hard_std = hard_subset.std()
            # Use the color from our palette for consistency
            ax.axhline(hard_val, color=custom_palette['HardCBM'], linestyle='--', label='HardCBM Baseline')
            ax.text(ax.get_xlim()[1], hard_val, f' {hard_val:.3f}', 
                    va='center', color=custom_palette['HardCBM'], fontweight='bold')
            ax.axhspan(hard_mean - hard_std, hard_mean + hard_std, 
                       color=custom_palette['HardCBM'], alpha=0.15, 
                       label='HardCBM ±1 SD')
                
            # 3. Add Optional Dashed lines for the bounds
            ax.axhline(hard_mean + hard_std, color=custom_palette['HardCBM'], 
                       linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(hard_mean - hard_std, color=custom_palette['HardCBM'], 
                       linestyle='--', alpha=0.5, linewidth=1)

        ax.set_title(f"{dataset_name}: {title}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Value")
        ax.set_xlabel(r"$\lambda_c$")
        
        if i == 0:
            ax.legend(title="Model", loc='best', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()