"""
Paired t-tests: GC-CBM vs Sequential CBM and Hard CBM
for CTL and ICL across TabularToy, dSprites, and CUB.

Values drawn from per-split _results.joblib files (computed during training).
Uses test_ctl_average / test_icl_average with NaN fallback to normalised variant.

Run from project root:
    python experiments/evaluate_models/statistical_tests_cbm_leakage.py
"""
import os, sys
import numpy as np
import joblib
from scipy import stats

sys.path.insert(0, os.getcwd())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_splits(folder, model_name, n_splits=5):
    """
    Load per-split results; return list of result dicts.
    Handles the dSprites Hard CBM quirk where splits 3-4 were saved with
    'lr1e-4' instead of 'lr1e-04' inside the same folder.
    """
    alt_name = model_name.replace("lr1e-04", "lr1e-4")
    model_dir = os.path.join(folder, model_name)
    results = []
    for i in range(n_splits):
        p = os.path.join(model_dir, f"{model_name}_split_{i}_results.joblib")
        if not os.path.exists(p) and alt_name != model_name:
            p = os.path.join(model_dir, f"{alt_name}_split_{i}_results.joblib")
        if os.path.exists(p):
            results.append(joblib.load(p))
    return results


def get_ctl(splits):
    vals = []
    for r in splits:
        v = r.get("test_ctl_average")
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = r.get("test_normalised_ctl")
        if v is not None and not np.isnan(float(v)):
            vals.append(float(v))
    return vals


def get_icl(splits):
    vals = []
    for r in splits:
        v = r.get("test_icl_average")
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = r.get("test_normalised_icl")
        if v is not None and not np.isnan(float(v)):
            # cap at 1.0 (same rule as paper_verify.py)
            vals.append(min(float(v), 1.0))
    return vals


def paired_ttest(a_vals, b_vals, a_name, b_name, metric, min_pairs=3):
    """Paired t-test on the shorter of the two lists (common folds)."""
    n = min(len(a_vals), len(b_vals))
    if n < min_pairs:
        print(f"      {metric}: insufficient paired folds ({n}) — skip")
        return
    a, b = np.array(a_vals[:n]), np.array(b_vals[:n])
    t, p = stats.ttest_rel(a, b)
    sig = " *" if p < 0.05 else ("  ." if p < 0.10 else "")
    direction = "lower" if np.mean(a) < np.mean(b) else "higher"
    print(
        f"      {metric:4s}  {a_name}: {np.mean(a):.4f}±{np.std(a):.4f}  "
        f"vs  {b_name}: {np.mean(b):.4f}±{np.std(b):.4f}  "
        f"| t={t:.2f}  p={p:.3f}{sig}  [{direction}]  n={n}"
    )


def run_dataset(ds_name, gc_cbm_specs, seq_spec, hard_spec):
    """
    gc_cbm_specs : list of (folder, model_name, lam_c_label)
    seq_spec     : (folder, model_name)
    hard_spec    : (folder, model_name)
    """
    print(f"\n{'='*70}")
    print(f"  {ds_name}")
    print(f"{'='*70}")

    seq_splits  = load_splits(*seq_spec)
    hard_splits = load_splits(*hard_spec)
    seq_ctl  = get_ctl(seq_splits)
    seq_icl  = get_icl(seq_splits)
    hard_ctl = get_ctl(hard_splits)
    hard_icl = get_icl(hard_splits)

    print(f"  Sequential CBM : {len(seq_splits)} folds, CTL mean={np.mean(seq_ctl):.4f}, ICL mean={np.mean(seq_icl):.4f}")
    print(f"  Hard CBM       : {len(hard_splits)} folds, CTL mean={np.mean(hard_ctl):.4f}, ICL mean={np.mean(hard_icl):.4f}")

    for folder, model_name, lam_label in gc_cbm_specs:
        gc_splits = load_splits(folder, model_name)
        if not gc_splits:
            print(f"\n  GC-CBM {lam_label}: no splits found — skip")
            continue
        gc_ctl = get_ctl(gc_splits)
        gc_icl = get_icl(gc_splits)
        print(f"\n  GC-CBM {lam_label}: {len(gc_splits)} folds, CTL mean={np.mean(gc_ctl):.4f}, ICL mean={np.mean(gc_icl):.4f}")

        print(f"    vs Sequential:")
        paired_ttest(gc_ctl, seq_ctl,  "GC-CBM", "Seq",  "CTL")
        paired_ttest(gc_icl, seq_icl,  "GC-CBM", "Seq",  "ICL")

        print(f"    vs Hard:")
        paired_ttest(gc_ctl, hard_ctl, "GC-CBM", "Hard", "CTL")
        paired_ttest(gc_icl, hard_icl, "GC-CBM", "Hard", "ICL")


# ---------------------------------------------------------------------------
# Dataset specs
# ---------------------------------------------------------------------------

BASE = os.getcwd() + "/results/"

run_dataset(
    "TabularToy",
    gc_cbm_specs=[
        (BASE + "tabulartoy_25_10k_models_acbm_shared_critic",
         "ACBM_adam_lr0.05_bs64_lam1_none_lam_c0.1_shared_critic", "lam_c0.1"),
        (BASE + "tabulartoy_25_10k_models_acbm_shared_critic",
         "ACBM_adam_lr0.05_bs64_lam1_none_lam_c0.5_shared_critic", "lam_c0.5"),
        (BASE + "tabulartoy_25_10k_models_acbm_shared_critic",
         "ACBM_adam_lr0.05_bs64_lam1_none_lam_c1_shared_critic",   "lam_c1"),
    ],
    seq_spec  = (BASE + "tabulartoy_25_10k_models_acbm_shared_critic",
                 "SeqCBM_adam_lr0.05_bs64_lam_c1"),
    hard_spec = (BASE + "tabulartoy_25_10k_models",
                 "HardCBM_adam_lr0.05_bs64_lam_c1"),
)

run_dataset(
    "dSprites",
    gc_cbm_specs=[
        (BASE + "dsprites_ACBM_shared_critic",
         "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c0.1_shared_critic", "lam_c0.1"),
        (BASE + "dsprites_ACBM_shared_critic",
         "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c0.5_shared_critic", "lam_c0.5"),
        (BASE + "dsprites_ACBM_shared_critic",
         "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c1_shared_critic",   "lam_c1"),
    ],
    seq_spec  = (BASE + "dsprites_sequential_acbm",
                 "SeqCBM_adam_lr1e-04_bs64_lam_c1"),
    hard_spec = (BASE + "dsprites_dep_0_models",
                 "HardCBM_adam_lr1e-04_bs64_lam_c1"),
)

run_dataset(
    "CUB",
    gc_cbm_specs=[
        (BASE + "cub_acbm_shared_critic",
         "ACBM_adam_lr1e-04_bs256_lam1_none_lam_c0.1_shared_critic", "lam_c0.1"),
        (BASE + "cub_acbm_shared_critic",
         "ACBM_adam_lr1e-04_bs256_lam1_none_lam_c0.5_shared_critic", "lam_c0.5"),
        (BASE + "cub_acbm_shared_critic",
         "ACBM_adam_lr1e-04_bs256_lam1_none_lam_c1_shared_critic",   "lam_c1"),
    ],
    seq_spec  = (BASE + "cub_acbm_shared_critic",
                 "SeqCBM_adam_lr1e-04_bs256_lam_c1"),
    hard_spec = (BASE + "cub_soft_0.01",
                 "HardCBM_adam_lr1e-04_bs256_lam_c1"),
)

print("\n* p<0.05   . p<0.10")
