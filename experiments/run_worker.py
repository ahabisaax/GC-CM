import argparse
import yaml
import copy
import sys
import os
import torch
import pytorch_lightning as pl

# --- 1. SET UP PATHS & IMPORTS ---
# Add the project root to the path so we can import our modules
# Assumes this script is in 'experiments/', so we go up one level
master_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, master_folder)

# Now we can import from the main script
from experiments import experiment_utils
from experiments.run_experiments import _generate_dataset_and_update_config

# Import your training functions
import xai_concept_leakage.train.training as training

# Import your models (to get the class definitions)
from xai_concept_leakage.models.cbm import ConceptBottleneckModel
from xai_concept_leakage.models.gc_cbm import GCConceptBottleneckModel
from xai_concept_leakage.models.gc_cbm import GCConceptBottleneckModel as AdversarialConceptBottleneckModel  # backwards compat
from xai_concept_leakage.models.gc_cbm import GCConceptBottleneckModel as CriticRegularisedConceptBottleneckModel  # backwards compat


# ... import any other models you use ...


def main_worker():
    parser = argparse.ArgumentParser(
        description="Worker script for CBM grid search."
    )
    parser.add_argument(
        "--config", "-c", required=True, help="YAML config file for all experiments."
    )
    parser.add_argument(
        "--task-id", "-t", required=True, type=int, help="SGE array task ID (1-indexed)."
    )
    parser.add_argument(
        "--output-dir", "-o", required=True, help="Top-level directory to save results."
    )
    parser.add_argument(
        "--force-cpu", action="store_true", help="Force CPU training."
    )
    args = parser.parse_args()

    task_index = args.task_id - 1  # Convert from 1-indexed to 0-indexed

    print(f"[Worker {args.task_id}] Loading config from: {args.config}")
    with open(args.config, "r") as f:
        loaded_config = yaml.load(f, Loader=yaml.FullLoader)

    # --- 2. Re-generate the full list of experiment configs ---
    shared_params = loaded_config.get("shared_params", {})
    all_run_configs = []

    if "runs" not in loaded_config:
        print(f"Error: 'runs' key not found in {args.config}", file=sys.stderr)
        sys.exit(1)

    for current_config in loaded_config["runs"]:
        trial_config = copy.deepcopy(shared_params)
        trial_config.update(current_config)
        # Use the correct, refactored function name
        for run_config in experiment_utils.generate_hyperatemer_configs(trial_config):
            all_run_configs.append(run_config)

    # --- 3. Select *only* the config for this task ID ---
    try:
        my_run_config = all_run_configs[task_index]
    except IndexError:
        print(f"ERROR: Task ID {args.task_id} is out of bounds for {len(all_run_configs)} total experiments.")
        sys.exit(1)

    print(f"[Worker {args.task_id}] My config is: {my_run_config['run_name']}")

    # --- 4. Set up remaining parameters for the training function ---
    result_dir = args.output_dir
    rerun = shared_params.get("rerun", True)  # Default to True for cluster jobs
    project_name = shared_params.get("project_name", "")
    split = 0  # Assuming we are just running split 0. Adjust as needed.

    # Set the unique run name
    run_name = my_run_config["run_name"]
    full_run_name = f"{run_name}_split_{split}"

    # --- 5. Create the specific sub-folder for this run (NEW) ---
    run_specific_dir = os.path.join(result_dir, full_run_name)
    os.makedirs(run_specific_dir, exist_ok=True)
    my_run_config["result_dir"] = run_specific_dir

    print(f"[Worker {args.task_id}] Results will be saved to: {run_specific_dir}")

    # --- 6. Generate the dataset for this config ---
    pl.seed_everything(42 + split)  # Ensure reproducibility
    (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        concept_map,
        intervened_groups,
        task_class_weights,
        acquisition_costs,
    ) = _generate_dataset_and_update_config(my_run_config)

    # --- 7. Determine the correct training function ---
    if my_run_config["architecture"] == "IndependentConceptBottleneckModel":
        train_fn = training.train_independent_model
    elif my_run_config["architecture"] == "SequentialConceptBottleneckModel":
        train_fn = training.train_sequential_model
    else:
        # Default to the end-to-end trainer
        train_fn = training.train_end_to_end_model

    accelerator = "gpu" if (not args.force_cpu) and (torch.cuda.is_available()) else "cpu"
    devices = "auto" if accelerator == "gpu" else 1

    print(f"[Worker {args.task_id}] Using accelerator: {accelerator}")

    # --- 8. Call the training function with the *single* config ---
    # This is the same call made inside run_experiments.py

    # Handle 'use_adversarial' flag
    use_adversarial = my_run_config.get("use_adversarial", False)

    train_args = {
        "run_name": run_name,
        "task_class_weights": task_class_weights,
        "accelerator": accelerator,
        "devices": devices,
        "n_concepts": my_run_config["n_concepts"],
        "n_tasks": my_run_config["n_tasks"],
        "config": my_run_config,
        "train_dl": train_dl,
        "val_dl": val_dl,
        "test_dl": test_dl,
        "split": split,
        "result_dir": run_specific_dir,  # Use the specific dir
        "rerun": rerun,
        "project_name": project_name,
        "seed": 42 + split,
        "imbalance": imbalance,
        "old_results": None,
        "gradient_clip_val": my_run_config.get("gradient_clip_val", 0),
        "single_frequency_epochs": shared_params.get("single_frequency_epochs", 0),
        "activation_freq": shared_params.get("activation_freq", 0)
    }

    if use_adversarial:
        train_args["use_adversarial"] = True
        train_args["adversarial_delay"] = my_run_config["adversarial_delay"]

    model, model_results = train_fn(**train_args)

    print(f"[Worker {args.task_id}] Training complete.")


if __name__ == "__main__":
    main_worker()