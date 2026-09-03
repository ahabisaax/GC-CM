"""
Concept incompleteness experiment: CEM vs GC-CEM on a Gaussian synthetic
dataset where only k_obs of K_TOTAL concepts are supervised.

Architecture mirrors the actual GCConceptEmbeddingModel (gc_cem.py):
  - Per-concept context generators producing 2*emb_size vectors (pos+neg)
  - c_pred = p * pos_emb + (1-p) * neg_emb   (sample-specific embeddings)
  - Shared critic: critic IS c2y (shared object)
  - GRL applied to c_pred directly — no residual subtraction
  - Single optimizer over all parameters

Metrics tracked per k_obs:
  - RTL, RCL   : task and inter-concept leakage in c_pred
  - task_acc   : accuracy of y prediction
  - c_acc      : accuracy of supervised concept predictions
  - int_curve  : task accuracy as 0..k_obs concepts are intervened
                 (intervention uses sample-specific context embeddings
                  with true concept label for mixing)

Run from project root:
    python experiments/evaluate_models/validate_concept_incompleteness.py
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

from xai_concept_leakage.metrics.leakage import compute_RTL_RCL

plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       9,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth":  0.75,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.22,
    "grid.linewidth":  0.5,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
K_TOTAL  = 15
D_IN     = 60       # 4 × K_TOTAL
EMB_SIZE = 16
N_TASKS  = 2
N_TOTAL  = 6000
N_TRAIN  = 5000
N_TEST   = N_TOTAL - N_TRAIN
N_SEEDS  = 8
EPOCHS   = 400
BATCH    = 256
LR       = 3e-3
LAMBDA_C = 1.0
LAMBDA_ADV = 0.5

K_OBS_VALUES = list(range(1, K_TOTAL + 1))

CACHE_PATH = f"results/cache/concept_incompleteness_K{K_TOTAL}_v3.npz"
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

C_CEM  = "#4575b4"
C_GCEM = "#d73027"

_rng = np.random.RandomState(42)
W_PROJ = torch.tensor(_rng.randn(K_TOTAL, D_IN).astype(np.float32))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def generate_dataset(n_samples, seed):
    rng   = np.random.RandomState(seed)
    c_all = rng.randint(0, 2, (n_samples, K_TOTAL)).astype(np.float32)
    y     = (c_all.sum(1) >= K_TOTAL / 2).astype(np.int64)
    x     = c_all @ W_PROJ.numpy() + rng.randn(n_samples, D_IN).astype(np.float32)
    return torch.tensor(x), torch.tensor(c_all), torch.tensor(y)


# ---------------------------------------------------------------------------
# GRL — matches GradientReversalFunction in gc_cem.py
# ---------------------------------------------------------------------------
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (lambda_,) = ctx.saved_tensors
        return -lambda_ * grad_output, None


def grad_reverse(x, alpha):
    lam = torch.tensor(alpha, dtype=x.dtype, device=x.device)
    return GradientReversalFunction.apply(x, lam)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _mlp(in_dim, out_dim, hidden=32):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


class TinyCEM(nn.Module):
    """
    Mirrors GCConceptEmbeddingModel forward pass:
      - Shared backbone (pre_concept_model)
      - Per-concept context generators → 2*emb_size (pos+neg concatenated)
      - Per-concept prob generators → scalar probability
      - c_pred = p * pos_emb + (1-p) * neg_emb
      - c2y(c_pred) → y
    """
    def __init__(self, k_obs):
        super().__init__()
        self.k_obs    = k_obs
        self.emb_size = EMB_SIZE

        # shared backbone
        self.pre_concept = nn.Sequential(
            nn.Linear(D_IN, 64), nn.ReLU()
        )
        # per-concept context generators (→ pos_emb || neg_emb)
        self.context_gens = nn.ModuleList([
            nn.Linear(64, 2 * EMB_SIZE) for _ in range(k_obs)
        ])
        # per-concept probability generators
        self.prob_gens = nn.ModuleList([
            nn.Linear(2 * EMB_SIZE, 1) for _ in range(k_obs)
        ])
        # task head
        self.c2y = _mlp(k_obs * EMB_SIZE, N_TASKS)

    def forward(self, x):
        pre_c    = self.pre_concept(x)           # (B, 64)
        contexts = []
        c_sem    = []
        for i in range(self.k_obs):
            ctx  = self.context_gens[i](pre_c)   # (B, 2*emb_size)
            prob = torch.sigmoid(self.prob_gens[i](ctx))  # (B, 1)
            contexts.append(ctx.unsqueeze(1))    # (B, 1, 2*emb_size)
            c_sem.append(prob)                   # (B, 1)

        contexts = torch.cat(contexts, dim=1)    # (B, k_obs, 2*emb_size)
        c_sem    = torch.cat(c_sem,    dim=-1)   # (B, k_obs)

        # mix pos/neg embeddings by concept probability
        p       = c_sem.unsqueeze(-1)                      # (B, k_obs, 1)
        pos_emb = contexts[:, :, :EMB_SIZE]               # (B, k_obs, emb_size)
        neg_emb = contexts[:, :, EMB_SIZE:]               # (B, k_obs, emb_size)
        c_pred  = p * pos_emb + (1 - p) * neg_emb        # (B, k_obs, emb_size)
        c_pred  = c_pred.view(x.size(0), -1)              # (B, k_obs*emb_size)

        y_hat   = self.c2y(c_pred)
        return c_sem, c_pred, y_hat, contexts


class TinyGCCEM(TinyCEM):
    """
    GC-CEM: shared critic (critic IS c2y), GRL on c_pred — no residuals.
    Matches GCConceptEmbeddingModel with shared_critic=True, loss_type='grl'.
    """
    def __init__(self, k_obs):
        super().__init__(k_obs)
        self.critic = self.c2y   # shared — same object, not a copy


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(ModelClass, k_obs, seed):
    torch.manual_seed(seed)
    x_all, c_all, y_all = generate_dataset(N_TOTAL, seed)

    x_tr  = x_all[:N_TRAIN]
    c_tr  = c_all[:N_TRAIN, :k_obs]
    y_tr  = y_all[:N_TRAIN]
    x_te  = x_all[N_TRAIN:]
    c_te  = c_all[N_TRAIN:, :k_obs]
    y_te  = y_all[N_TRAIN:]
    c_te_all = c_all[N_TRAIN:]          # all K_TOTAL concepts for intervention

    dl = DataLoader(TensorDataset(x_tr, c_tr, y_tr), batch_size=BATCH, shuffle=True)

    model = ModelClass(k_obs)
    # single optimizer over ALL parameters (matches gc_cem.py GRL configure_optimizers)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    is_gc = isinstance(model, TinyGCCEM)

    model.train()
    for _ in range(EPOCHS):
        for xb, cb, yb in dl:
            opt.zero_grad()
            c_sem, c_pred, y_hat, _ = model(xb)

            task_loss    = F.cross_entropy(y_hat, yb)
            concept_loss = LAMBDA_C * F.binary_cross_entropy(c_sem, cb)
            loss         = task_loss + concept_loss

            if is_gc:
                # GRL on c_pred — same object used for task head (shared critic)
                grl_c_pred = grad_reverse(c_pred, LAMBDA_ADV)
                y_adv      = model.critic(grl_c_pred)
                loss       = loss + F.cross_entropy(y_adv, yb)

            loss.backward()
            opt.step()
        sched.step()

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        c_sem_te, c_pred_te, y_hat_te, ctx_te = model(x_te)
        c_sem_tr, c_pred_tr, _,        _      = model(x_tr)

    task_acc    = (y_hat_te.argmax(-1) == y_te).float().mean().item()
    concept_acc = ((c_sem_te > 0.5).float() == c_te).float().mean().item()

    # RTL / RCL on test embeddings (c_pred is k_obs*emb_size flat)
    # reshape to (N, k_obs, emb_size) for compute_RTL_RCL
    c_pred_te_3d = c_pred_te.reshape(len(x_te), k_obs, EMB_SIZE)
    c_pred_tr_3d = c_pred_tr.reshape(len(x_tr), k_obs, EMB_SIZE)
    r = compute_RTL_RCL(
        c_pred_tr_3d, c_pred_te_3d,
        c_all[:N_TRAIN, :k_obs], c_te,
        y_all[:N_TRAIN], y_te,
        global_norm=True,
    )
    rtl = r["RTL_sum"] / EMB_SIZE
    rcl = r["RCL_sum"] / EMB_SIZE

    # ── Intervention curve ────────────────────────────────────────────────────
    # Intervene on first n_int concepts using sample-specific context embeddings
    # with the TRUE concept label for mixing (mirrors _after_interventions in gc_cem.py)
    int_curve = [task_acc]
    with torch.no_grad():
        pos_te = ctx_te[:, :, :EMB_SIZE]   # (N_test, k_obs, emb_size)
        neg_te = ctx_te[:, :, EMB_SIZE:]

        for n_int in range(1, k_obs + 1):
            c_pred_int = c_pred_te.clone().reshape(len(x_te), k_obs, EMB_SIZE)
            c_true_int = c_te_all[:, :n_int]          # true binary labels
            c_true_exp = c_true_int.unsqueeze(-1)      # (N, n_int, 1)
            # replace first n_int embeddings with ground-truth prototype
            c_pred_int[:, :n_int] = (
                c_true_exp * pos_te[:, :n_int] +
                (1 - c_true_exp) * neg_te[:, :n_int]
            )
            y_int = model.c2y(c_pred_int.view(len(x_te), -1))
            int_curve.append((y_int.argmax(-1) == y_te).float().mean().item())

    return task_acc, concept_acc, rtl, rcl, np.array(int_curve)


# ---------------------------------------------------------------------------
# Main: train or load from cache
# ---------------------------------------------------------------------------
if os.path.exists(CACHE_PATH):
    print(f"Loading cached results from {CACHE_PATH}")
    _d = np.load(CACHE_PATH, allow_pickle=True)
    cem_rtl       = _d["cem_rtl"];      cem_rtl_std  = _d["cem_rtl_std"]
    cem_rcl       = _d["cem_rcl"];      cem_rcl_std  = _d["cem_rcl_std"]
    cem_task      = _d["cem_task"];     cem_cacc     = _d["cem_cacc"]
    gc_rtl        = _d["gc_rtl"];       gc_rtl_std   = _d["gc_rtl_std"]
    gc_rcl        = _d["gc_rcl"];       gc_rcl_std   = _d["gc_rcl_std"]
    gc_task       = _d["gc_task"];      gc_cacc      = _d["gc_cacc"]
    cem_int_curves = list(_d["cem_int_curves"])
    gc_int_curves  = list(_d["gc_int_curves"])
else:
    print(f"Training on Gaussian toy (K_TOTAL={K_TOTAL}, D={D_IN}, m={EMB_SIZE})")
    print("Architecture: per-concept context generators, shared critic = c2y, GRL on c_pred\n")

    cem_rtl_v, cem_rtl_std_v = [], []
    cem_rcl_v, cem_rcl_std_v = [], []
    cem_task_v, cem_cacc_v   = [], []
    cem_int_curves            = []

    gc_rtl_v,  gc_rtl_std_v  = [], []
    gc_rcl_v,  gc_rcl_std_v  = [], []
    gc_task_v,  gc_cacc_v    = [], []
    gc_int_curves             = []

    for k_obs in K_OBS_VALUES:
        print(f"--- k_obs={k_obs}/{K_TOTAL} ---")
        for prefix, ModelClass, rtl_v, rcl_v, ta_v, ca_v, ic_v in [
            ("CEM",    TinyCEM,   cem_rtl_v, cem_rcl_v, cem_task_v, cem_cacc_v, []),
            ("GC-CEM", TinyGCCEM, gc_rtl_v,  gc_rcl_v,  gc_task_v,  gc_cacc_v,  []),
        ]:
            seed_rtl, seed_rcl, seed_ta, seed_ca, seed_ic = [], [], [], [], []
            for seed in range(N_SEEDS):
                print(f"  {prefix} seed={seed}", end="  ", flush=True)
                ta, ca, rtl, rcl, ic = train_model(ModelClass, k_obs, seed)
                seed_rtl.append(rtl); seed_rcl.append(rcl)
                seed_ta.append(ta);   seed_ca.append(ca)
                seed_ic.append(ic)
                print(f"task={ta:.3f}  cacc={ca:.3f}  int_full={ic[-1]:.3f}"
                      f"  RTL={rtl:.4f}  RCL={rcl:.4f}")

            rtl_v.append(np.mean(seed_rtl))
            rcl_v.append(np.mean(seed_rcl))
            ta_v.append(np.mean(seed_ta))
            ca_v.append(np.mean(seed_ca))
            ic_v.append(np.mean(seed_ic, axis=0))   # mean curve over seeds
            if prefix == "CEM":
                cem_rtl_std_v.append(np.std(seed_rtl))
                cem_rcl_std_v.append(np.std(seed_rcl))
                cem_int_curves.append(np.mean(seed_ic, axis=0))
            else:
                gc_rtl_std_v.append(np.std(seed_rtl))
                gc_rcl_std_v.append(np.std(seed_rcl))
                gc_int_curves.append(np.mean(seed_ic, axis=0))

    def _a(l): return np.array(l)
    cem_rtl, cem_rtl_std = _a(cem_rtl_v), _a(cem_rtl_std_v)
    cem_rcl, cem_rcl_std = _a(cem_rcl_v), _a(cem_rcl_std_v)
    cem_task, cem_cacc   = _a(cem_task_v), _a(cem_cacc_v)
    gc_rtl,  gc_rtl_std  = _a(gc_rtl_v),  _a(gc_rtl_std_v)
    gc_rcl,  gc_rcl_std  = _a(gc_rcl_v),  _a(gc_rcl_std_v)
    gc_task, gc_cacc     = _a(gc_task_v),  _a(gc_cacc_v)

    np.savez(CACHE_PATH,
             cem_rtl=cem_rtl, cem_rtl_std=cem_rtl_std,
             cem_rcl=cem_rcl, cem_rcl_std=cem_rcl_std,
             cem_task=cem_task, cem_cacc=cem_cacc,
             gc_rtl=gc_rtl,   gc_rtl_std=gc_rtl_std,
             gc_rcl=gc_rcl,   gc_rcl_std=gc_rcl_std,
             gc_task=gc_task,  gc_cacc=gc_cacc,
             cem_int_curves=np.array(cem_int_curves, dtype=object),
             gc_int_curves=np.array(gc_int_curves,   dtype=object))
    print(f"\nCache saved → {CACHE_PATH}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
PLOT_DIR = "results/plots/cbm/"
os.makedirs(PLOT_DIR, exist_ok=True)
xs = np.array(K_OBS_VALUES)
N_SEEDS_LOADED = 8
se = lambda std: std / np.sqrt(N_SEEDS_LOADED)

# ── Figure 1: summary panels ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(13, 2.8))

cem_int_final = np.array([c[-1] for c in cem_int_curves])
gc_int_final  = np.array([c[-1] for c in gc_int_curves])

panels = [
    (axes[0], cem_rtl, cem_rtl_std, gc_rtl, gc_rtl_std, "RTL",                   "(a)"),
    (axes[1], cem_rcl, cem_rcl_std, gc_rcl, gc_rcl_std, "RCL",                   "(b)"),
    (axes[2], cem_task, None,        gc_task, None,      "Task accuracy",          "(c)"),
    (axes[3], cem_int_final, None,   gc_int_final, None, "Full intervention acc.", "(d)"),
]

for ax, cm, cs, gm, gs, ylabel, lbl in panels:
    ax.plot(xs, cm, 'o-', color=C_CEM,  label="CEM",    lw=1.9, ms=4)
    ax.plot(xs, gm, 's-', color=C_GCEM, label="GC-CEM", lw=1.9, ms=4)
    if cs is not None:
        ax.fill_between(xs, cm - se(cs), cm + se(cs), alpha=0.18, color=C_CEM)
        ax.fill_between(xs, gm - se(gs), gm + se(gs), alpha=0.18, color=C_GCEM)
    ax.set_xlabel("$k_{obs}$ (supervised concepts)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, K_TOTAL + 1, 2))
    ax.text(0.03, 0.97, lbl, transform=ax.transAxes, va='top',
            fontsize=9, fontweight='bold')
    if ylabel in ("RTL", "RCL"): ax.set_ylim(bottom=0)
    if "acc" in ylabel:          ax.set_ylim(bottom=0.5)

axes[0].legend(loc='upper right', framealpha=0.8)
plt.tight_layout()
out1 = PLOT_DIR + f"paper_concept_incompleteness_K{K_TOTAL}.png"
plt.savefig(out1); plt.close()
print(f"Saved → {out1}")

# ── Figure 2: full intervention curves at 4 representative k_obs ──────────────
rep_ks = [4, 8, 11, 14]
fig2, axes2 = plt.subplots(2, 4, figsize=(12, 4.5), sharey=True)

for col, k in enumerate(rep_ks):
    i = K_OBS_VALUES.index(k)
    for row, (arr, label, color) in enumerate([
        (cem_int_curves[i], "CEM",    C_CEM),
        (gc_int_curves[i],  "GC-CEM", C_GCEM),
    ]):
        ax = axes2[row, col]
        xs_int = np.arange(len(arr))
        ax.plot(xs_int, arr, '-', color=color, lw=2)
        ax.axhline(arr[0], color='gray', lw=0.8, ls=':', alpha=0.6)
        ax.set_title(f"$k_{{obs}}={k}$", fontsize=9)
        ax.set_xlabel("# interventions")
        ax.set_xticks(range(0, len(arr), max(1, len(arr) // 4)))
        ax.set_ylim(0.5, 1.02)
        if col == 0: ax.set_ylabel(f"{label}\nTask acc")

plt.suptitle(
    f"Intervention curves — K_TOTAL={K_TOTAL}, shared critic, GRL on $c_{{pred}}$",
    fontsize=10, y=1.01,
)
plt.tight_layout()
out2 = PLOT_DIR + f"paper_concept_incompleteness_int_curves_K{K_TOTAL}.png"
plt.savefig(out2); plt.close()
print(f"Saved → {out2}")

# ── Summary table ──────────────────────────────────────────────────────────────
print(f"\n{'k':>4}  {'CEM RTL':>9}  {'GC RTL':>9}  "
      f"{'CEM RCL':>9}  {'GC RCL':>9}  {'CEM int':>8}  {'GC int':>8}  {'Δint':>6}")
print("-" * 90)
for i, k in enumerate(K_OBS_VALUES):
    print(f"  {k:2d}  {cem_rtl[i]:9.4f}  {gc_rtl[i]:9.4f}  "
          f"{cem_rcl[i]:9.4f}  {gc_rcl[i]:9.4f}  "
          f"{cem_int_final[i]:8.3f}  {gc_int_final[i]:8.3f}  "
          f"{gc_int_final[i]-cem_int_final[i]:+6.3f}")

print("\nDone.")
