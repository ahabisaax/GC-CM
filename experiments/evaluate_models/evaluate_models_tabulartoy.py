import os
import sys
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='pkg_resources')
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', message='.*UnsupportedFieldAttributeWarning.*')
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

master_folder = os.getcwd().replace("/experiments/evaluate_models", "")
sys.path.insert(0, master_folder)
master_folder = master_folder + "/"

from xai_concept_leakage.data.tabulartoy_auxiliary import TT_dataloaders
from experiments.experiment_utils import get_tabulartoy_extractor_arch
from experiments.evaluate_model import evaluate_CBM
from experiments.fuse_models_folds import model_names_fold_from_classes

if __name__ == "__main__":

    ##################################################################################################
    ### Config:
    ##################################################################################################
    # Automatically discover all model folders in tabulartoy_25_10k_models_acem_shared_critic
    results_folder = master_folder + "results/tabulartoy_25_10k_models_acem_shared_critic/"
    
    # Get all model subdirectories
    model_dirs = [d for d in os.listdir(results_folder) 
                  if os.path.isdir(os.path.join(results_folder, d)) and d != 'auto']
    
    print(f"Found {len(model_dirs)} model directories:")
    for model_dir in model_dirs:
        print(f"  - {model_dir}")
    
    # Generate checkpoint names for all models with their folds
    checkpoint_names = []
    for model_dir in model_dirs:
        # Find all .pt files in the directory
        model_path = os.path.join(results_folder, model_dir)
        pt_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
        # Extract checkpoint names (with .pt extension)
        for pt_file in pt_files:
            checkpoint_names.append(os.path.join(model_dir, pt_file))
    
    print(f"\nTotal checkpoints to evaluate: {len(checkpoint_names)}")
    
    save_path = master_folder + "results/results_tabulartoy_25_10k_crcem.dict"
    repeats = 5
    rerun = True

    test_dl_label = "test"
    policies = ["random", "coop"]
    ois_repeats = repeats
    intervention_repeats = repeats
    interconcept_repeats = repeats
    concepts_task_repeats = repeats

    # Only evaluate CEM_leakage_scores
    list_observables = ["CEM_leakage_scores"]

    ##################################################################################################
    ### Dataloading:
    ##################################################################################################
    data_folder = master_folder + "data/TabularToy/"
    cov = 0.25
    n_samples = 10000
    considered_concepts = ["0", "1", "2"]
    save_folder = (
        data_folder
        + "tabulartoy_"
        + str(int(cov * 100))
        + "_"
        + str(int(n_samples // 1000))
        + "k/"
    )

    train_dl, val_dl, test_dl = TT_dataloaders(
        save_folder, considered_concepts=considered_concepts, c_logits=False
    )
    dl_dict = {"train": train_dl, "val": val_dl, "test": test_dl}
    x2c_extractor = get_tabulartoy_extractor_arch

    ##################################################################################################
    ### Evaluate:
    ##################################################################################################

    results = evaluate_CBM(
        checkpoint_names=checkpoint_names,
        results_folder=results_folder,
        x2c_extractor=x2c_extractor,
        dl_dict=dl_dict,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        list_observables=list_observables,
        ois_repeats=ois_repeats,
        policies=policies,
        intervention_repeats=intervention_repeats,
        interconcept_repeats=interconcept_repeats,
        concepts_task_repeats=concepts_task_repeats,
        rerun=rerun,
        save_path=save_path,
    )
