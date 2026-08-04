"""
Signal injection calibration for real CEM embeddings.

Question: How much synthetic task-signal must be injected into real CUB CEM
embeddings before CVL and CTL become detectably non-zero?

This quantifies each metric's detection threshold given the actual "other"
variance present in the CUB CEM embedding space.

Procedure
---------
1. Load real CUB CEM embeddings  c_mix[N, K, m]  and  c_true[N, K], y[N].
2. For each concept k, compute the mean embedding per class:
       gamma_k[j] = mean(c_mix[y==j, k, :])           for j in 0..C-1
   (This is the empirical class centroid in embedding space.)
3. Inject signal of amplitude alpha into the embedding:
       c_inj[n, k, :] = c_mix[n, k, :] + alpha * (gamma_k[y[n]] - mu_k)
   where mu_k = mean(gamma_k) centres the injected signal.
4. Estimate CVL and CTL on the injected embeddings across a grid of alpha.
5. Find alpha_CVL and alpha_CTL = smallest alpha at which the metric first
   exceeds 3 * sigma_0 (where sigma_0 is the std at alpha=0, estimated over
   repeated subsamples).

The injected gamma is built from the same samples, so it is not "new"
leakage — it amplifies any residual structure already present in the real
class means. At alpha=0 we recover the real embeddings. As alpha→∞ the
embedding is dominated by class-label structure and both metrics should
saturate.

Run with:
    conda run -n xai-leakage python experiments/evaluate_models/signal_injection_calibration.py [--dataset cub|dsprites|tabulartoy]
"""
import argparse
import os
import sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)

from xai_concept_leakage.metrics.mutual_information import compute_mi_matrix_parallel

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="cub", choices=["cub", "dsprites", "tabulartoy"])
parser.add_argument("--fold", type=int, default=1)
parser.add_argument("--n_alpha", type=int, default=12,
                    help="number of alpha values to sweep (log scale 0.01..32)")
parser.add_argument("--n_bootstrap", type=int, default=5,
                    help="bootstrap resamples for estimating metric std at alpha=0")
parser.add_argument("--n_neighbors", type=int, default=3)
parser.add_argument("--max_ksg_samples", type=int, default=2000)
parser.add_argument("--skip_ctl", action="store_true",
                    help="skip KSG CTL (fast mode; only CVL computed)")
parser.add_argument("--alpha_min", type=float, default=0.0)
parser.add_argument("--alpha_max", type=float, default=32.0)
args = parser.parse_args()

SAVE_PATH = master + f"results/results_signal_injection_{args.dataset}.dict"
n_jobs = os.cpu_count() or 1

# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------
def load_cub(fold):
    sys.path.insert(0, os.path.join(master, "data/CUB200"))
    import data.CUB200.cub_loader as cub
    cfg = {
        "dataset": "cub", "num_workers": 0, "batch_size": 256,
        "root_dir": os.path.join(master, "data/CUB200/"),
        "sampling_percent": 1, "sampling_groups": True,
        "test_subsampling": 1, "weight_loss": True, "train_augment": False,
    }
    _, _, test_dl, _, _ = cub.generate_data(config=cfg, seed=42,
                                             output_dataset_vars=True,
                                             root_dir=cfg["root_dir"])
    from experiments.evaluate_models.eval_suite import load_predictions
    pt = (master + f"results/cub_cem/CEM_adam_lr1e-03_bs256_lam_c0.1/"
          f"CEM_adam_lr1e-03_bs256_lam_c0.1_fold_{fold}.pt")
    return load_predictions(pt, None, test_dl)


def load_dsprites(fold):
    from xai_concept_leakage.data.dsprites_auxiliary import dsprites_dataloaders
    from experiments.experiment_utils import get_dsprites_extractor_arch
    from experiments.evaluate_models.eval_suite import load_predictions
    _, _, test_dl = dsprites_dataloaders(
        master + "data/dsprites/dsprites_dep_0.npz", batch_size=64, num_workers=0
    )
    pt = (master + f"results/dsprites_cem/CEM_adam_lr1e-03_bs64_lam_c0.1/"
          f"CEM_adam_lr1e-03_bs64_lam_c0.1_fold_{fold}.pt")
    return load_predictions(pt, get_dsprites_extractor_arch, test_dl)


def load_tabulartoy(fold):
    from xai_concept_leakage.data.tabulartoy_loader import generate_data as tt_gen
    from experiments.experiment_utils import get_tabulartoy_extractor_arch
    from experiments.evaluate_models.eval_suite import load_predictions
    cfg = {"dataset": "tabulartoy", "batch_size": 512, "num_workers": 0,
           "root_dir": master + "data/TabularToy/tabulartoy_25_10k/",
           "dependency_param": 0.25, "test_subsampling": 1}
    _, _, test_dl, _, _ = tt_gen(cfg, seed=42, output_dataset_vars=True)
    pt = (master + f"results/tabulartoy_25_10k_models_acem_shared_critic/"
          f"CEM_adam_lr0.05_bs64_lam_c0.1/"
          f"CEM_adam_lr0.05_bs64_lam_c0.1_fold_{fold}.pt")
    return load_predictions(pt, get_tabulartoy_extractor_arch, test_dl)


LOADERS = {"cub": load_cub, "dsprites": load_dsprites, "tabulartoy": load_tabulartoy}

# ---------------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------------
print(f"Loading {args.dataset} CEM embeddings (fold={args.fold}) ...", flush=True)
preds = LOADERS[args.dataset](args.fold)

c_mix  = preds["c_mix"]   # [N, K, m]
c_true = preds["c_true"]  # [N, K]
y_true = preds["y_true"]  # [N]

N, K, m = c_mix.shape
C = int(y_true.max()) + 1
print(f"  N={N}, K={K}, m={m}, C={C}")

# ---------------------------------------------------------------------------
# Build empirical class centroids for each concept
# ---------------------------------------------------------------------------
print("Computing empirical class centroids ...", flush=True)
gamma = np.zeros((K, C, m))  # gamma[k, j, :] = mean embedding for concept k, class j
for k in range(K):
    for j in range(C):
        mask = y_true == j
        if mask.sum() > 0:
            gamma[k, j, :] = c_mix[mask, k, :].mean(axis=0)
        # else stays zero — rare classes

mu = gamma.mean(axis=1, keepdims=True)  # [K, 1, m]
gamma_centred = gamma - mu              # [K, C, m]

# ---------------------------------------------------------------------------
# Metric estimators
# ---------------------------------------------------------------------------
def estimate_cvl(c_emb, c_tr, y_tr, c_te, y_te):
    """Ridge CVL on test set (train/test split assumed already done)."""
    Y_tr = y_tr.reshape(-1, 1).astype(float)
    Y_te = y_te.reshape(-1, 1).astype(float)
    cvl_vals = []
    for k in range(K):
        # Step 1: regress concept label out
        reg1 = Ridge(alpha=1.0)
        reg1.fit(c_tr[:, k:k+1], c_emb["train"][:, k, :])
        R_tr = c_emb["train"][:, k, :] - reg1.predict(c_tr[:, k:k+1])
        R_te = c_emb["test"][:, k, :]  - reg1.predict(c_te[:, k:k+1])
        # Step 2: regress task label onto residual
        reg2 = Ridge(alpha=1.0)
        reg2.fit(Y_tr, R_tr)
        cvl_vals.append(max(0.0,
            r2_score(R_te, reg2.predict(Y_te), multioutput="variance_weighted")))
    return float(np.mean(cvl_vals))


def estimate_ctl(c_emb_flat, c_tr_flat, y_tr):
    """KSG CTL on a subsample."""
    N_ = len(y_tr)
    e_f = c_emb_flat
    c_f = c_tr_flat
    y_  = y_tr
    if N_ > args.max_ksg_samples:
        idx = np.random.choice(N_, args.max_ksg_samples, replace=False)
        e_f, c_f, y_ = e_f[idx], c_f[idx], y_[idx]
    mi_e = compute_mi_matrix_parallel(
        e_f, d=y_, n_neighbors=args.n_neighbors,
        normalise=True, flatten=False, n_jobs=n_jobs)
    mi_c = compute_mi_matrix_parallel(
        c_f, d=y_, n_neighbors=args.n_neighbors,
        normalise=True, flatten=False, n_jobs=n_jobs)
    return float(np.maximum(0, np.mean(mi_e) - np.mean(mi_c)))


# ---------------------------------------------------------------------------
# Train/test split (80/20)
# ---------------------------------------------------------------------------
rng = np.random.RandomState(42)
idx = rng.permutation(N)
n_tr = int(0.8 * N)
tr_idx, te_idx = idx[:n_tr], idx[n_tr:]

def split(arr):
    return arr[tr_idx], arr[te_idx]

c_mix_tr, c_mix_te   = split(c_mix)
c_true_tr, c_true_te = split(c_true)
y_tr,      y_te       = split(y_true)

# ---------------------------------------------------------------------------
# Alpha sweep
# ---------------------------------------------------------------------------
alpha_vals = [0.0] + list(np.logspace(
    np.log10(max(args.alpha_min, 1e-3)),
    np.log10(args.alpha_max),
    args.n_alpha - 1,
))

results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}
key_prefix = f"{args.dataset}_fold{args.fold}"

print(f"\nSweeping alpha ∈ {[round(a,3) for a in alpha_vals]}")
print(f"  skip_ctl={args.skip_ctl}  n_bootstrap={args.n_bootstrap}  n_jobs={n_jobs}\n")

for alpha in alpha_vals:
    key = f"{key_prefix}_alpha{alpha:.4f}"
    if key in results:
        r = results[key]
        msg = (f"alpha={alpha:.4f}  CVL={r['cvl_mean']:.4f}±{r['cvl_std']:.4f}")
        if not args.skip_ctl:
            msg += f"  CTL={r['ctl_mean']:.4f}±{r['ctl_std']:.4f}"
        print(msg + "  [cached]")
        continue

    # Inject signal
    def inject(c_emb, y):
        """Add alpha * gamma_centred[k, y[n], :] to c_emb[n, k, :]."""
        out = c_emb.copy()
        for k in range(K):
            out[:, k, :] += alpha * gamma_centred[k, y, :]
        return out

    cvl_vals, ctl_vals = [], []
    for _ in range(args.n_bootstrap):
        # Resample training set (bootstrap; test set fixed)
        boot_idx = rng.choice(n_tr, n_tr, replace=True)
        c_emb_b_tr = inject(c_mix_tr[boot_idx], y_tr[boot_idx])
        c_emb_te   = inject(c_mix_te, y_te)

        emb_split = {"train": c_emb_b_tr, "test": c_emb_te}
        cvl_vals.append(estimate_cvl(
            emb_split, c_true_tr[boot_idx], y_tr[boot_idx], c_true_te, y_te
        ))
        if not args.skip_ctl:
            e_flat = c_emb_b_tr.reshape(len(boot_idx), -1)
            c_flat = c_true_tr[boot_idx]
            ctl_vals.append(estimate_ctl(e_flat, c_flat, y_tr[boot_idx]))

    entry = {
        "cvl_mean": float(np.mean(cvl_vals)),
        "cvl_std":  float(np.std(cvl_vals)),
        "ctl_mean": float(np.mean(ctl_vals)) if ctl_vals else None,
        "ctl_std":  float(np.std(ctl_vals))  if ctl_vals else None,
    }
    results[key] = entry
    joblib.dump(results, SAVE_PATH)

    msg = f"alpha={alpha:.4f}  CVL={entry['cvl_mean']:.4f}±{entry['cvl_std']:.4f}"
    if not args.skip_ctl:
        msg += f"  CTL={entry['ctl_mean']:.4f}±{entry['ctl_std']:.4f}"
    print(msg, flush=True)

# ---------------------------------------------------------------------------
# Detection threshold: 3-sigma above alpha=0
# ---------------------------------------------------------------------------
def detection_threshold(metric_key_mean, metric_key_std):
    a0_key = f"{key_prefix}_alpha0.0000"
    if a0_key not in results:
        return None
    baseline_mean = results[a0_key][metric_key_mean]
    baseline_std  = results[a0_key][metric_key_std]
    threshold = baseline_mean + 3 * baseline_std
    for alpha in alpha_vals[1:]:
        key = f"{key_prefix}_alpha{alpha:.4f}"
        if key in results:
            val = results[key][metric_key_mean]
            if val is not None and val >= threshold:
                return alpha
    return None

alpha_cvl = detection_threshold("cvl_mean", "cvl_std")
alpha_ctl = detection_threshold("ctl_mean", "ctl_std") if not args.skip_ctl else None
print(f"\nDetection thresholds (3σ above baseline):")
print(f"  CVL: alpha = {alpha_cvl}")
if not args.skip_ctl:
    print(f"  CTL: alpha = {alpha_ctl}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
print("\nGenerating plot ...", flush=True)

present_alphas = [a for a in alpha_vals if f"{key_prefix}_alpha{a:.4f}" in results]
cvl_m = np.array([results[f"{key_prefix}_alpha{a:.4f}"]["cvl_mean"] for a in present_alphas])
cvl_s = np.array([results[f"{key_prefix}_alpha{a:.4f}"]["cvl_std"]  for a in present_alphas])
if not args.skip_ctl:
    ctl_m = np.array([results[f"{key_prefix}_alpha{a:.4f}"]["ctl_mean"] or 0 for a in present_alphas])
    ctl_s = np.array([results[f"{key_prefix}_alpha{a:.4f}"]["ctl_std"]  or 0 for a in present_alphas])

n_panels = 1 if args.skip_ctl else 2
fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))
if n_panels == 1:
    axes = [axes]

BLUE, GREEN = "#2166ac", "#1a9641"

# CVL panel
ax = axes[0]
ax.plot(present_alphas, cvl_m, "o-", color=BLUE, linewidth=2, label="CVL (mean)")
ax.fill_between(present_alphas, cvl_m - cvl_s, cvl_m + cvl_s,
                color=BLUE, alpha=0.25, linewidth=0)
# Baseline ± 3σ
a0k = f"{key_prefix}_alpha0.0000"
if a0k in results:
    b_mean = results[a0k]["cvl_mean"]
    b_std  = results[a0k]["cvl_std"]
    ax.axhline(b_mean + 3 * b_std, color=BLUE, linestyle="--", linewidth=1.2,
               label="baseline + 3σ")
    ax.axhline(b_mean, color=BLUE, linestyle=":", linewidth=1.0, alpha=0.5)
if alpha_cvl is not None:
    ax.axvline(alpha_cvl, color="grey", linestyle="--", linewidth=1.2,
               label=f"detection threshold α={alpha_cvl:.3f}")
ax.set_xscale("symlog", linthresh=0.01)
ax.set_xlabel("Injection amplitude α", fontsize=11)
ax.set_ylabel("CVL", fontsize=11)
ax.set_ylim(bottom=0)
ax.legend(fontsize=9)
ax.set_title(f"CVL signal injection calibration\n({args.dataset} CEM fold {args.fold})",
             fontsize=10, fontweight="bold")

# CTL panel
if not args.skip_ctl:
    ax = axes[1]
    ax.plot(present_alphas, ctl_m, "D-", color=GREEN, linewidth=2, label="CTL (mean)")
    ax.fill_between(present_alphas, ctl_m - ctl_s, ctl_m + ctl_s,
                    color=GREEN, alpha=0.25, linewidth=0)
    if a0k in results and results[a0k]["ctl_mean"] is not None:
        b_mean = results[a0k]["ctl_mean"]
        b_std  = results[a0k]["ctl_std"]
        ax.axhline(b_mean + 3 * b_std, color=GREEN, linestyle="--", linewidth=1.2,
                   label="baseline + 3σ")
        ax.axhline(b_mean, color=GREEN, linestyle=":", linewidth=1.0, alpha=0.5)
    if alpha_ctl is not None:
        ax.axvline(alpha_ctl, color="grey", linestyle="--", linewidth=1.2,
                   label=f"detection threshold α={alpha_ctl:.3f}")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("Injection amplitude α", fontsize=11)
    ax.set_ylabel("CTL (KSG)", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.set_title(f"CTL signal injection calibration\n({args.dataset} CEM fold {args.fold})",
                 fontsize=10, fontweight="bold")

plt.tight_layout()
out_base = master + f"experiments/evaluate_models/signal_injection_{args.dataset}"
for ext in (".pdf", ".png"):
    plt.savefig(out_base + ext, bbox_inches="tight",
                dpi=150 if ext == ".png" else None)
    print(f"Saved → {out_base + ext}")
