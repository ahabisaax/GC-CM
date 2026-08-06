"""
Cache CEM and GC-CEM concept embeddings for all lambda_c values and all folds,
for TabularToy, dSprites, and CUB.

Uses the clean-split protocol: pool val+test, shuffle, split N_TR/N_TE.

Usage:
    python experiments/evaluate_models/cache_embeddings_all.py --dataset tabulartoy
    python experiments/evaluate_models/cache_embeddings_all.py --dataset dsprites
    python experiments/evaluate_models/cache_embeddings_all.py --dataset cub

Output (one file per dataset):
    results/embeddings_tabulartoy_all.joblib
    results/embeddings_dsprites_all.joblib
    results/embeddings_cub_all.joblib

Cache format:
    {
        "<MODEL>_<lam_c>_<fold>": (c_mix_tr, c_true_tr, y_tr,
                                    c_mix_te, c_true_te, y_te),
        ...
    }
    where MODEL is "CEM" or "GC-CEM", e.g. "CEM_lam_c0.1_fold_1"
"""
import argparse, os, sys
import numpy as np
import joblib

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)
sys.path.insert(0, os.path.join(master, "data/CUB200"))

from experiments.evaluate_models.eval_suite import (
    load_predictions, ALL_FOLDS, find_checkpoints,
)

SEED    = 42
LAM_C_LIST = ["lam_c0.1", "lam_c0.5", "lam_c1"]

# Clean-split sizes per dataset
SPLIT_SIZES = {
    "tabulartoy": (2000, 900),   # pool val(2000)+test(1000) → tr=2100 te=900
    "dsprites":   (5000, 1000),
    "cub":        (5992, 1000),
}


def pool_and_split(preds_a, preds_b, n_tr, n_te, seed=SEED):
    """Pool two prediction dicts (val+test), shuffle, split into CVL train/test."""
    c_mix  = np.concatenate([preds_a["c_mix"],   preds_b["c_mix"]],  axis=0)
    c_true = np.concatenate([preds_a["c_true"],  preds_b["c_true"]], axis=0)
    y      = np.concatenate([preds_a["y_true"],  preds_b["y_true"]], axis=0)

    N   = len(y)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)
    n_tr_actual = min(n_tr, N - n_te)
    tr_idx = idx[:n_tr_actual]
    te_idx = idx[n_tr_actual:n_tr_actual + n_te]

    print(f"    pooled {N} → train={n_tr_actual}  test={n_te}  "
          f"c_mix={c_mix.shape}  C={int(y.max())+1}", flush=True)
    return (
        c_mix[tr_idx].astype(np.float32),
        c_true[tr_idx].astype(np.float32),
        y[tr_idx].astype(np.int64),
        c_mix[te_idx].astype(np.float32),
        c_true[te_idx].astype(np.float32),
        y[te_idx].astype(np.int64),
    )


# ---------------------------------------------------------------------------
# Dataset-specific setup
# ---------------------------------------------------------------------------

def setup_tabulartoy():
    from xai_concept_leakage.data.tabulartoy_auxiliary import TT_dataloaders
    from experiments.experiment_utils import get_tabulartoy_extractor_arch

    data_folder = master + "data/TabularToy/tabulartoy_25_10k/"
    train_dl, val_dl, test_dl = TT_dataloaders(
        data_folder, considered_concepts=["0", "1", "2"],
        c_logits=False, num_workers=0,
    )
    x2c = get_tabulartoy_extractor_arch

    cem_folder  = master + "results/tabulartoy_25_10k_models_acem_shared_critic/"
    gcem_folder = cem_folder  # same folder

    model_specs = [
        ("CEM",    cem_folder,  "CEM_"),
        ("GC-CEM", gcem_folder, "ACEM_"),
    ]
    return val_dl, test_dl, x2c, model_specs


def setup_dsprites():
    from xai_concept_leakage.data.dsprites_auxiliary import dsprites_dataloaders
    from experiments.experiment_utils import get_dsprites_extractor_arch

    path = master + "data/dsprites/dsprites_dep_0.npz"
    train_dl, val_dl, test_dl = dsprites_dataloaders(
        path, val_ratio=0.1, num_workers=0, batch_size=256,
    )
    x2c = get_dsprites_extractor_arch

    cem_folder  = master + "results/dsprites_cem/"
    gcem_folder = master + "results/dsprites_acem_shared_critic/"

    model_specs = [
        ("CEM",    cem_folder,  "CEM_"),
        ("GC-CEM", gcem_folder, "CRCEM_"),
    ]
    return val_dl, test_dl, x2c, model_specs


def setup_cub():
    import data.CUB200.cub_loader as cub

    cfg = {
        "dataset": "cub", "num_workers": 0, "batch_size": 256,
        "root_dir": master + "data/CUB200/",
        "sampling_percent": 1, "sampling_groups": True,
        "test_subsampling": 1, "weight_loss": True, "train_augment": False,
    }
    _, val_dl, test_dl, _, _ = cub.generate_data(
        config=cfg, seed=SEED, output_dataset_vars=True,
        root_dir=master + "data/CUB200/",
    )
    x2c = None  # CUB model is self-contained

    cem_folder  = master + "results/cub_cem/"
    gcem_folder = cem_folder  # same folder

    model_specs = [
        ("CEM",    cem_folder,  "CEM_"),
        ("GC-CEM", gcem_folder, "CRCEM_"),
    ]
    return val_dl, test_dl, x2c, model_specs


SETUP_FN = {
    "tabulartoy": setup_tabulartoy,
    "dsprites":   setup_dsprites,
    "cub":        setup_cub,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["tabulartoy", "dsprites", "cub"])
    ap.add_argument("--folds", nargs="+", default=ALL_FOLDS)
    ap.add_argument("--lam_c", nargs="+", default=LAM_C_LIST)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    out_path = args.output or (master + f"results/embeddings_{args.dataset}_all.joblib")
    n_tr, n_te = SPLIT_SIZES[args.dataset]

    print(f"Dataset : {args.dataset}")
    print(f"λ_c     : {args.lam_c}")
    print(f"Folds   : {args.folds}")
    print(f"Output  : {out_path}")
    print(f"Split   : train={n_tr}  test={n_te}\n", flush=True)

    print(f"Loading {args.dataset} dataloaders ...", flush=True)
    val_dl, test_dl, x2c, model_specs = SETUP_FN[args.dataset]()
    print(f"  val={len(val_dl.dataset)}  test={len(test_dl.dataset)}\n", flush=True)

    cache = joblib.load(out_path) if os.path.exists(out_path) else {}

    for lam_c in args.lam_c:
        for model_label, folder, prefix in model_specs:
            try:
                ckpts = find_checkpoints(folder, prefix, lam_c, args.folds)
            except AssertionError as e:
                print(f"  SKIP {model_label} {lam_c}: {e}", flush=True)
                continue

            for fold, ckpt in ckpts.items():
                key = f"{model_label}_{lam_c}_{fold}"
                if key in cache:
                    print(f"  Already cached: {key}", flush=True)
                    continue

                print(f"\n  Extracting {key}", flush=True)
                print(f"    ckpt: {os.path.relpath(ckpt, master)}", flush=True)

                preds_val  = load_predictions(ckpt, x2c, val_dl)
                preds_test = load_predictions(ckpt, x2c, test_dl)

                cache[key] = pool_and_split(preds_val, preds_test, n_tr, n_te)
                joblib.dump(cache, out_path)
                print(f"    saved → {os.path.relpath(out_path, master)}", flush=True)

    print(f"\nDone. {len(cache)} entries in cache.")
    for k in sorted(cache.keys()):
        shapes = [arr.shape for arr in cache[k]]
        print(f"  {k}: c_mix_tr={shapes[0]}  y_tr={shapes[2]}  "
              f"c_mix_te={shapes[3]}  y_te={shapes[5]}")


if __name__ == "__main__":
    main()
