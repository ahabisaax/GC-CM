"""
Standalone script: generate prototype alignment plots for TabularToy and dSprites.

Runs load_predictions on fold_1 of each model and saves:
  results/plots/tabulartoy/prototype_alignment.png
  results/plots/dsprites/prototype_alignment.png

No metric caching — pure visualisation only.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")

master_folder = os.getcwd().replace("/experiments/evaluate_models", "")
sys.path.insert(0, master_folder)
master_folder = master_folder + "/"

from experiments.evaluate_models.eval_suite import (
    find_checkpoints,
    load_predictions,
    prototype_alignment_plot,
)

LAM_C = "lam_c0.1"
FOLD  = "fold_1"


def run_tabulartoy():
    print("=" * 60)
    print("TabularToy")
    print("=" * 60)

    from xai_concept_leakage.data.tabulartoy_auxiliary import TT_dataloaders
    from experiments.experiment_utils import get_tabulartoy_extractor_arch

    data_folder = master_folder + "data/TabularToy/"
    save_folder = data_folder + "tabulartoy_25_10k/"
    _, _, test_dl = TT_dataloaders(
        save_folder, considered_concepts=["0", "1", "2"], c_logits=False, num_workers=1
    )

    results_folder = master_folder + "results/tabulartoy_25_10k_models_acem_shared_critic/"
    cem_ckpts   = find_checkpoints(results_folder, "CEM_",   LAM_C, [FOLD])
    crcem_ckpts = find_checkpoints(results_folder, "CRCEM_", LAM_C, [FOLD])

    if not (cem_ckpts and crcem_ckpts):
        print("  No checkpoints found — skipping TabularToy")
        return

    print(f"  CEM   checkpoint: {os.path.relpath(cem_ckpts[FOLD], master_folder)}")
    print(f"  CRCEM checkpoint: {os.path.relpath(crcem_ckpts[FOLD], master_folder)}")
    print("  Loading CEM predictions...")
    cem_preds   = load_predictions(cem_ckpts[FOLD],   get_tabulartoy_extractor_arch, test_dl)
    print("  Loading CRCEM predictions...")
    crcem_preds = load_predictions(crcem_ckpts[FOLD], get_tabulartoy_extractor_arch, test_dl)

    plot_dir = master_folder + "results/plots/tabulartoy/"
    os.makedirs(plot_dir, exist_ok=True)

    scores = prototype_alignment_plot(
        cem_preds["c_pos"],   cem_preds["c_neg"],
        crcem_preds["c_pos"], crcem_preds["c_neg"],
        label_a="CEM", label_b="CRCEM",
        concept_names=[f"c{k}" for k in range(cem_preds["n_concepts"])],
        title=f"TabularToy — prototype alignment ({FOLD})",
        save_path=plot_dir + "prototype_alignment.png",
    )
    print(f"  CEM   pos-neg dist={scores['CEM']['pos_neg_dist']:.4f}"
          f"  concept sep={scores['CEM']['concept_sep']:.4f}")
    print(f"  CRCEM pos-neg dist={scores['CRCEM']['pos_neg_dist']:.4f}"
          f"  concept sep={scores['CRCEM']['concept_sep']:.4f}")


def run_dsprites():
    print()
    print("=" * 60)
    print("dSprites")
    print("=" * 60)

    try:
        from xai_concept_leakage.data.dsprites_auxiliary import dsprites_dataloaders
        from experiments.experiment_utils import get_dsprites_extractor_arch
    except ImportError as e:
        print(f"  Skipping dSprites — missing dependency: {e}")
        return

    path_dataset = master_folder + "data/dsprites/dsprites_dep_0.npz"
    _, _, test_dl = dsprites_dataloaders(path_dataset, val_ratio=0.1, num_workers=1, batch_size=256)

    cem_ckpts   = find_checkpoints(master_folder + "results/dsprites_cem/",                "CEM_",   LAM_C, [FOLD])
    crcem_ckpts = find_checkpoints(master_folder + "results/dsprites_acem_shared_critic/", "CRCEM_", LAM_C, [FOLD])

    if not (cem_ckpts and crcem_ckpts):
        print("  No checkpoints found — skipping dSprites")
        return

    print(f"  CEM   checkpoint: {os.path.relpath(cem_ckpts[FOLD], master_folder)}")
    print(f"  CRCEM checkpoint: {os.path.relpath(crcem_ckpts[FOLD], master_folder)}")
    print("  Loading CEM predictions...")
    cem_preds   = load_predictions(cem_ckpts[FOLD],   get_dsprites_extractor_arch, test_dl)
    print("  Loading CRCEM predictions...")
    crcem_preds = load_predictions(crcem_ckpts[FOLD], get_dsprites_extractor_arch, test_dl)

    plot_dir = master_folder + "results/plots/dsprites/"
    os.makedirs(plot_dir, exist_ok=True)

    scores = prototype_alignment_plot(
        cem_preds["c_pos"],   cem_preds["c_neg"],
        crcem_preds["c_pos"], crcem_preds["c_neg"],
        label_a="CEM", label_b="CRCEM",
        concept_names=[f"c{k}" for k in range(cem_preds["n_concepts"])],
        title=f"dSprites — prototype alignment ({FOLD})",
        save_path=plot_dir + "prototype_alignment.png",
    )
    print(f"  CEM   pos-neg dist={scores['CEM']['pos_neg_dist']:.4f}"
          f"  concept sep={scores['CEM']['concept_sep']:.4f}")
    print(f"  CRCEM pos-neg dist={scores['CRCEM']['pos_neg_dist']:.4f}"
          f"  concept sep={scores['CRCEM']['concept_sep']:.4f}")


if __name__ == "__main__":
    run_tabulartoy()
    run_dsprites()
