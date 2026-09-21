"""
Train tiny CEM models on the Gaussian generative model at varying λ_c,
then evaluate CVL and ICVL on the learned embeddings at test time.

This validates that leakage emerges naturally from training dynamics — not
by construction — and that CVL/ICVL correctly recover it.

Generative model
----------------
  c_k ~ Bernoulli(0.5)  for k = 1..K   (independent binary concepts)
  x   = c @ W_proj + ε                  (noisy linear observation of concepts)
  y   = 1[sum(c) >= K/2]               (majority vote task)

CEM architecture
----------------
  x2c : MLP(x → K logits)  →  p_k = sigmoid(logit_k)
  emb : ĉ_k = p_k * e_k^+  +  (1-p_k) * e_k^-   (learnable pos/neg prototypes)
  c2y : Linear(K * emb_size → n_tasks)

Loss:
  L = λ_c * mean_k BCE(p_k, c_k)  +  CE(y_hat, y)

Leakage mechanism: the task loss back-propagates through the e_k^± prototypes,
encoding task information directly into the embedding geometry.  With low λ_c
the concept term is weak so task shortcutting is unconstrained — CVL and ICVL
should be high.  With high λ_c the concept supervision dominates and the
embeddings are anchored to concept labels, suppressing leakage.

Run from project root:
    python experiments/evaluate_models/validate_trained_cem_lambda_c.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from xai_concept_leakage.metrics.leakage import (
    compute_CVL, compute_ICVL, compute_RTL_RCL,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "axes.linewidth":     0.75,
    "lines.linewidth":    1.8,
    "lines.markersize":   5.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.22,
    "grid.linewidth":     0.5,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

# ---------------------------------------------------------------------------
# Data parameters
# ---------------------------------------------------------------------------
N_TOTAL   = 6000
N_TRAIN   = 5000
N_TEST    = N_TOTAL - N_TRAIN
K         = 6       # annotated concepts
K_TOTAL   = 7       # K annotated + 1 latent (c_extra, never supervised)
D_IN      = 24      # input dimension
EMB_SIZE  = 16      # CEM embedding dimension per concept
N_TASKS   = 2       # binary task

# Concept incompleteness: c_extra is a real concept that determines y
# alongside c_1..c_6, but is NEVER annotated.  The model must encode it
# somewhere in the embeddings to achieve maximum task accuracy — this is
# the core leakage mechanism, matching real datasets (CUB, dSprites) where
# the annotated concepts don't fully explain the task label.
# γ controls how clearly c_extra can be decoded from x.
GAMMA_EXTRA = 4.0

# Fixed random projections
_rng42    = np.random.RandomState(42)
W_PROJ    = torch.tensor(_rng42.randn(K_TOTAL, D_IN).astype(np.float32))  # all K+1 concepts → x

# ---------------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------------
LAMBDA_C_LIST = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
LAM_EXPERIMENTAL = {0.1, 0.5, 1.0}
N_SEEDS   = 10
EPOCHS    = 400
BATCH     = 256
LR        = 3e-3

C_CVL   = "#0072B2"
C_ICVL  = "#009E73"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def generate_dataset(n_samples, seed):
    rng = np.random.RandomState(seed)
    # All K+1 concepts (c_extra is the last one — latent, never annotated)
    c_all  = rng.randint(0, 2, (n_samples, K_TOTAL)).astype(np.float32)
    c_obs  = c_all[:, :K]    # K annotated concepts — what the model is supervised on
    c_extra = c_all[:, K]    # latent concept — determines y but is never given to model

    # Task: majority vote over ALL K+1 concepts including the latent one.
    # From annotated concepts alone, the model can't fully predict y when
    # the vote is tied at K/2 — c_extra breaks the tie.
    y = (c_all.sum(axis=1) >= (K_TOTAL / 2)).astype(np.int64)

    # x encodes all K+1 concepts (concept projection) + noise.
    # c_extra gets an extra amplitude boost (GAMMA_EXTRA) so the model
    # CAN decode it from x if it chooses to — the information is there.
    c_for_x      = c_all.copy()
    c_for_x[:, K] *= GAMMA_EXTRA    # amplify c_extra signal in x
    x = (c_for_x @ W_PROJ.numpy()
         + rng.randn(n_samples, D_IN).astype(np.float32))

    return (
        torch.tensor(x),
        torch.tensor(c_obs),   # only K annotated labels returned
        torch.tensor(y),
    )


# ---------------------------------------------------------------------------
# Minimal CEM
# ---------------------------------------------------------------------------
class TinyCEM(nn.Module):
    def __init__(self, input_dim, n_concepts, emb_size, n_tasks):
        super().__init__()
        # Small x2c — limited capacity so concept prediction is imperfect,
        # creating pressure to encode extra task signal in the embeddings.
        self.x2c = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_concepts),
        )
        # Learnable positive and negative prototype vectors per concept
        self.emb_pos = nn.Parameter(torch.randn(n_concepts, emb_size) * 0.1)
        self.emb_neg = nn.Parameter(torch.randn(n_concepts, emb_size) * 0.1)
        # c2y: small MLP — enough capacity to exploit embedding structure
        self.c2y = nn.Sequential(
            nn.Linear(n_concepts * emb_size, 32),
            nn.ReLU(),
            nn.Linear(32, n_tasks),
        )

        self.n_concepts = n_concepts
        self.emb_size   = emb_size

    def forward(self, x):
        logits = self.x2c(x)                              # [N, K]
        p      = torch.sigmoid(logits)                    # [N, K]
        # Concept embeddings: ĉ_k = p_k * e_k^+ + (1-p_k) * e_k^-
        p_exp  = p.unsqueeze(-1)                          # [N, K, 1]
        c_hat  = p_exp * self.emb_pos + (1 - p_exp) * self.emb_neg  # [N, K, m]
        c_flat = c_hat.reshape(x.size(0), -1)             # [N, K*m]
        y_hat  = self.c2y(c_flat)                         # [N, n_tasks]  # noqa
        return y_hat, p, c_hat


def train_cem(lambda_c, seed):
    torch.manual_seed(seed)
    x_all, c_all, y_all = generate_dataset(N_TOTAL, seed)

    x_tr, c_tr, y_tr = x_all[:N_TRAIN], c_all[:N_TRAIN], y_all[:N_TRAIN]
    x_te, c_te, y_te = x_all[N_TRAIN:], c_all[N_TRAIN:], y_all[N_TRAIN:]

    train_ds = TensorDataset(x_tr, c_tr, y_tr)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

    model = TinyCEM(D_IN, K, EMB_SIZE, N_TASKS)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    model.train()
    for epoch in range(EPOCHS):
        for xb, cb, yb in train_dl:
            opt.zero_grad()
            y_hat, p, _ = model(xb)
            task_loss    = F.cross_entropy(y_hat, yb)
            concept_loss = F.binary_cross_entropy(p, cb)
            loss = task_loss + lambda_c * concept_loss
            loss.backward()
            opt.step()
        sched.step()

    # --- Test-time evaluation ---
    model.eval()
    with torch.no_grad():
        y_hat_te, p_te, c_hat_te = model(x_te)

    task_acc   = (y_hat_te.argmax(-1) == y_te).float().mean().item()
    concept_acc = ((p_te > 0.5).float() == c_te).float().mean().item()

    # Convert to numpy for CVL / ICVL
    c_hat_tr_np = None
    with torch.no_grad():
        _, _, c_hat_tr_t = model(x_tr)
    c_hat_tr_np = c_hat_tr_t.numpy()      # [N_train, K, m]
    c_hat_te_np = c_hat_te.numpy()        # [N_test,  K, m]
    c_tr_np     = c_tr.numpy()
    c_te_np     = c_te.numpy()
    y_tr_np     = y_tr.numpy()
    y_te_np     = y_te.numpy()

    return c_hat_tr_np, c_hat_te_np, c_tr_np, c_te_np, y_tr_np, y_te_np, task_acc, concept_acc


# ---------------------------------------------------------------------------
# Sweep λ_c  (cached — rerun skips training if cache exists)
# ---------------------------------------------------------------------------
CACHE_PATH = "results/cache/trained_cem_lambda_c_rtl.npz"
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

if os.path.exists(CACHE_PATH):
    print(f"Loading cached results from {CACHE_PATH}")
    _d = np.load(CACHE_PATH)
    cvl_means   = _d["cvl_means"];   cvl_stds   = _d["cvl_stds"]
    icvl_means  = _d["icvl_means"];  icvl_stds  = _d["icvl_stds"]
    focal_means = _d["focal_means"]; focal_stds = _d["focal_stds"]
    rtl_means   = _d["rtl_means"];   rtl_stds   = _d["rtl_stds"]
    rcl_means   = _d["rcl_means"];   rcl_stds   = _d["rcl_stds"]
    task_accs   = _d["task_accs"];   conc_accs  = _d["conc_accs"]
else:
    print(f"Training TinyCEM on Gaussian toy  "
          f"(K={K}, D={D_IN}, m={EMB_SIZE}, N={N_TRAIN}+{N_TEST})\n")

    cvl_means,   cvl_stds   = [], []
    icvl_means,  icvl_stds  = [], []
    focal_means, focal_stds = [], []
    rtl_means,   rtl_stds   = [], []
    rcl_means,   rcl_stds   = [], []
    task_accs,   conc_accs  = [], []

    for lam in LAMBDA_C_LIST:
        cvl_vals, icvl_vals, focal_vals = [], [], []
        rtl_vals, rcl_vals = [], []
        ta_vals, ca_vals = [], []

        for seed in range(N_SEEDS):
            print(f"  λ_c={lam:.2f}  seed={seed}", end="  ", flush=True)
            result = train_cem(lam, seed)
            c_hat_tr, c_hat_te, c_tr, c_te, y_tr, y_te, ta, ca = result

            r_cvl  = compute_CVL(c_hat_tr, c_hat_te, c_tr, c_te, y_tr, y_te)
            cvl_vals.append(r_cvl["CVL"])

            r_icvl = compute_ICVL(c_hat_tr, c_hat_te, c_tr, c_te)
            icvl_vals.append(r_icvl["ICVL"])
            focal_vals.append(np.array(r_icvl["ICVL_matrix"])[1, 0])

            # Canonical leakage metrics (RTL_norm = RTL_sum / d, paper defn).
            # global_norm=True matches the setting used for every reported
            # dataset number in results_rtl_rcl_all_datasets.dict.
            r_res = compute_RTL_RCL(
                c_hat_tr, c_hat_te, c_tr, c_te, y_tr, y_te, global_norm=True)
            rtl_vals.append(r_res["RTL_norm"])
            rcl_vals.append(r_res["RCL_norm"])

            ta_vals.append(ta); ca_vals.append(ca)
            print(f"task={ta:.3f}  concept={ca:.3f}  CVL={r_cvl['CVL']:.4f}"
                  f"  RTL={r_res['RTL_norm']:.4f}  RCL={r_res['RCL_norm']:.4f}")

        cvl_means.append(np.mean(cvl_vals));   cvl_stds.append(np.std(cvl_vals))
        icvl_means.append(np.mean(icvl_vals)); icvl_stds.append(np.std(icvl_vals))
        focal_means.append(np.mean(focal_vals)); focal_stds.append(np.std(focal_vals))
        rtl_means.append(np.mean(rtl_vals));   rtl_stds.append(np.std(rtl_vals))
        rcl_means.append(np.mean(rcl_vals));   rcl_stds.append(np.std(rcl_vals))
        task_accs.append(np.mean(ta_vals));    conc_accs.append(np.mean(ca_vals))
        print(f"  → CVL={cvl_means[-1]:.4f} ± {cvl_stds[-1]:.4f}"
              f"  ICVL={icvl_means[-1]:.4f} ± {icvl_stds[-1]:.4f}"
              f"  RTL={rtl_means[-1]:.4f} ± {rtl_stds[-1]:.4f}"
              f"  RCL={rcl_means[-1]:.4f} ± {rcl_stds[-1]:.4f}\n")

    cvl_means   = np.array(cvl_means);   cvl_stds   = np.array(cvl_stds)
    icvl_means  = np.array(icvl_means);  icvl_stds  = np.array(icvl_stds)
    focal_means = np.array(focal_means); focal_stds = np.array(focal_stds)
    rtl_means   = np.array(rtl_means);   rtl_stds   = np.array(rtl_stds)
    rcl_means   = np.array(rcl_means);   rcl_stds   = np.array(rcl_stds)
    task_accs   = np.array(task_accs);   conc_accs  = np.array(conc_accs)

    np.savez(CACHE_PATH,
             cvl_means=cvl_means,   cvl_stds=cvl_stds,
             icvl_means=icvl_means, icvl_stds=icvl_stds,
             focal_means=focal_means, focal_stds=focal_stds,
             rtl_means=rtl_means,   rtl_stds=rtl_stds,
             rcl_means=rcl_means,   rcl_stds=rcl_stds,
             task_accs=task_accs,   conc_accs=conc_accs)
    print(f"Cache saved → {CACHE_PATH}")

xs = np.array(LAMBDA_C_LIST)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

panel_data = [
    (axes[0], cvl_means,  cvl_stds,  C_CVL,  "CVL",             "(a)"),
    (axes[1], icvl_means, icvl_stds, C_ICVL, "ICVL (aggregate)", "(b)"),
]

for ax, means, stds, color, ylabel, letter in panel_data:
    ax.plot(xs, means, color=color, lw=1.9, marker="o", ms=5.5, zorder=3)
    ax.fill_between(xs, means - stds, means + stds,
                    color=color, alpha=0.15, zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel(r"Concept supervision weight $\lambda_c$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(xs[0] * 0.6, xs[-1] * 1.6)
    ax.set_ylim(bottom=0, top=means.max() * 1.18)

    # Clean y-axis: 4 evenly-spaced ticks from 0 to max
    ymax = means.max()
    yticks = np.linspace(0, ymax, 4)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=7.5)
    ax.text(-0.18, 1.10, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

fig.tight_layout(w_pad=1.8)
fig.subplots_adjust(top=0.92)

PLOT_DIR = "results/plots/cbm/"
os.makedirs(PLOT_DIR, exist_ok=True)
out = PLOT_DIR + "paper_trained_cem_lambda_c.pdf"
fig.savefig(out)
fig.savefig(out.replace(".pdf", ".png"))
print(f"\nSaved → {out}")

# ---------------------------------------------------------------------------
# Plot 2: the canonical metrics (RTL_norm / RCL_norm = sum / d)
#
# This is the figure to cite for λ_c, not validate_lambda_c_rtl.py — that one
# injects leakage analytically as alpha/λ_c, so its curve falls by
# construction. Here λ_c only changes the training objective.
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(6.8, 2.8))

for ax, means, stds, color, ylabel, letter in [
    (axes2[0], rtl_means, rtl_stds, C_CVL,  "RTL",  "(a)"),
    (axes2[1], rcl_means, rcl_stds, C_ICVL, "RCL",  "(b)"),
]:
    ax.plot(xs, means, color=color, lw=1.9, marker="o", ms=5.5, zorder=3)
    ax.fill_between(xs, means - stds, means + stds,
                    color=color, alpha=0.15, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"Concept supervision weight $\lambda_c$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(xs[0] * 0.6, xs[-1] * 1.6)
    ax.set_ylim(bottom=0, top=max(means.max() * 1.18, 1e-6))
    ax.set_yticks(np.linspace(0, means.max(), 4))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=7.5)
    ax.text(-0.18, 1.10, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

fig2.tight_layout(w_pad=1.8)
fig2.subplots_adjust(top=0.92)
out2 = PLOT_DIR + "paper_trained_cem_lambda_c_rtl_rcl.pdf"
fig2.savefig(out2)
fig2.savefig(out2.replace(".pdf", ".png"))
print(f"Saved → {out2}")

# Summary table
print(f"\n{'λ_c':>6}  {'RTL':>8}  {'RCL':>8}  {'CVL':>8}  {'ICVL_focal':>12}  "
      f"{'task_acc':>10}  {'conc_acc':>10}")
for i, lam in enumerate(LAMBDA_C_LIST):
    print(f"{lam:>6.2f}  {rtl_means[i]:>8.4f}  {rcl_means[i]:>8.4f}  "
          f"{cvl_means[i]:>8.4f}  {focal_means[i]:>12.4f}  "
          f"{task_accs[i]:>10.4f}  {conc_accs[i]:>10.4f}")
plt.close(fig)
print("Done.")
