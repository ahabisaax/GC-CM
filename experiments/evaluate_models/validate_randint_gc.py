"""
RandInt + GC training experiment: does random-intervention training during
GC-CEM training improve intervention performance?

Three conditions on a complete Gaussian toy (K=20 concepts, all supervised):
  1. CEM          — no adversarial, no randint
  2. GC-CEM       — adversarial (GRL), no randint
  3. GC-CEM+RInt  — adversarial (GRL) + random intervention training

Architecture mirrors gc_cem.py (shared critic = c2y, GRL on c_pred, no residuals).

Run from project root:
    python experiments/evaluate_models/validate_randint_gc.py
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
K_TOTAL   = 20
D_IN      = 80        # 4 × K_TOTAL
EMB_SIZE  = 16
N_TASKS   = 2
N_TOTAL   = 6000
N_TRAIN   = 5000
N_TEST    = N_TOTAL - N_TRAIN
N_SEEDS   = 8
EPOCHS    = 400
BATCH     = 256
LR        = 3e-3
LAMBDA_C  = 1.0
LAMBDA_ADV = 0.5
P_RANDINT  = 0.25     # probability per concept of being intervened during training

CACHE_PATH = "results/cache/randint_gc_K20_noisy.npz"
PLOT_DIR   = "results/plots/cbm/"
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

C_CEM   = "#4575b4"
C_GC    = "#d73027"
C_GCRI  = "#1a9850"

_rng  = np.random.RandomState(42)
W_PROJ = torch.tensor(_rng.randn(K_TOTAL, D_IN).astype(np.float32))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
NOISE_SCALE = 4.0   # input noise std; pushes concept acc to ~80-85%

def generate_dataset(n_samples, seed):
    rng   = np.random.RandomState(seed)
    c_all = rng.randint(0, 2, (n_samples, K_TOTAL)).astype(np.float32)
    y     = (c_all.sum(1) >= K_TOTAL / 2).astype(np.int64)
    x     = c_all @ W_PROJ.numpy() + NOISE_SCALE * rng.randn(n_samples, D_IN).astype(np.float32)
    return torch.tensor(x), torch.tensor(c_all), torch.tensor(y)


# ---------------------------------------------------------------------------
# GRL
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
# Model
# ---------------------------------------------------------------------------
def _mlp(in_dim, out_dim, hidden=64):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


class TinyCEM(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_concept  = nn.Sequential(nn.Linear(D_IN, 64), nn.ReLU())
        self.context_gens = nn.ModuleList([nn.Linear(64, 2 * EMB_SIZE) for _ in range(K_TOTAL)])
        self.prob_gens    = nn.ModuleList([nn.Linear(2 * EMB_SIZE, 1)  for _ in range(K_TOTAL)])
        self.c2y          = _mlp(K_TOTAL * EMB_SIZE, N_TASKS)

    def encode(self, x):
        """Returns c_sem (B, K), c_pred (B, K, emb), contexts (B, K, 2*emb)."""
        pre_c = self.pre_concept(x)
        ctxs, probs = [], []
        for i in range(K_TOTAL):
            ctx  = self.context_gens[i](pre_c)
            prob = torch.sigmoid(self.prob_gens[i](ctx))
            ctxs.append(ctx.unsqueeze(1))
            probs.append(prob)
        contexts = torch.cat(ctxs,  dim=1)          # (B, K, 2*emb)
        c_sem    = torch.cat(probs, dim=-1)          # (B, K)
        p        = c_sem.unsqueeze(-1)               # (B, K, 1)
        pos_emb  = contexts[:, :, :EMB_SIZE]
        neg_emb  = contexts[:, :, EMB_SIZE:]
        c_pred   = p * pos_emb + (1 - p) * neg_emb  # (B, K, emb)
        return c_sem, c_pred, contexts

    def forward(self, x):
        c_sem, c_pred, contexts = self.encode(x)
        y_hat = self.c2y(c_pred.view(x.size(0), -1))
        return c_sem, c_pred, y_hat, contexts


class TinyGCCEM(TinyCEM):
    def __init__(self):
        super().__init__()
        self.critic = self.c2y   # shared object — same as gc_cem.py shared_critic=True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_cem(seed):
    return _train(seed, use_gc=False, use_randint=False)

def train_cem_randint(seed):
    return _train(seed, use_gc=False, use_randint=True)

def train_gccm(seed):
    return _train(seed, use_gc=True, use_randint=False)

def train_gccm_randint(seed):
    return _train(seed, use_gc=True, use_randint=True)


def _train(seed, use_gc, use_randint):
    torch.manual_seed(seed)
    x_all, c_all, y_all = generate_dataset(N_TOTAL, seed)
    x_tr, c_tr, y_tr = x_all[:N_TRAIN], c_all[:N_TRAIN], y_all[:N_TRAIN]
    x_te, c_te, y_te = x_all[N_TRAIN:], c_all[N_TRAIN:], y_all[N_TRAIN:]

    dl    = DataLoader(TensorDataset(x_tr, c_tr, y_tr), batch_size=BATCH, shuffle=True)
    model = TinyGCCEM() if use_gc else TinyCEM()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    model.train()
    for _ in range(EPOCHS):
        for xb, cb, yb in dl:
            opt.zero_grad()
            c_sem, c_pred, contexts = model.encode(xb)

            # randint: replace probabilities with c_true before mixing, matching
            # _after_interventions in cem.py (prob * (1-mask) + mask * c_true),
            # so context generators still receive gradients for intervened concepts
            if use_randint:
                mask = torch.bernoulli(torch.ones(K_TOTAL) * P_RANDINT)       # (K,)
                mask = mask.unsqueeze(0).expand(xb.size(0), -1)               # (B, K)
                probs_int = c_sem * (1 - mask) + mask * cb                    # replace p with c_true
                pos = contexts[:, :, :EMB_SIZE]
                neg = contexts[:, :, EMB_SIZE:]
                c_pred = (probs_int.unsqueeze(-1) * pos +
                          (1 - probs_int).unsqueeze(-1) * neg)                # (B, K, emb)

            c_flat    = c_pred.view(xb.size(0), -1)
            y_hat     = model.c2y(c_flat)
            task_loss = F.cross_entropy(y_hat, yb)
            conc_loss = LAMBDA_C * F.binary_cross_entropy(c_sem, cb)
            loss      = task_loss + conc_loss

            if use_gc:
                grl_c   = grad_reverse(c_flat, LAMBDA_ADV)
                y_adv   = model.critic(grl_c)
                loss    = loss + F.cross_entropy(y_adv, yb)

            loss.backward()
            opt.step()
        sched.step()

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        c_sem_te, c_pred_te, y_hat_te, ctx_te = model(x_te)

    task_acc  = (y_hat_te.argmax(-1) == y_te).float().mean().item()
    conc_acc  = ((c_sem_te > 0.5).float() == c_te).float().mean().item()

    # ── Intervention curve (0..K interventions) ───────────────────────────────
    int_curve = [task_acc]
    with torch.no_grad():
        pos_te = ctx_te[:, :, :EMB_SIZE]
        neg_te = ctx_te[:, :, EMB_SIZE:]
        for n_int in range(1, K_TOTAL + 1):
            c_int   = c_pred_te.clone()
            c_true  = c_te[:, :n_int].unsqueeze(-1)
            c_int[:, :n_int] = c_true * pos_te[:, :n_int] + (1 - c_true) * neg_te[:, :n_int]
            y_int   = model.c2y(c_int.view(len(x_te), -1))
            int_curve.append((y_int.argmax(-1) == y_te).float().mean().item())

    return task_acc, conc_acc, np.array(int_curve)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if os.path.exists(CACHE_PATH):
    print(f"Loading cache from {CACHE_PATH}")
    d = np.load(CACHE_PATH, allow_pickle=True)
    cem_curves   = d["cem_curves"]
    gc_curves    = d["gc_curves"]
    gcri_curves  = d["gcri_curves"]
    cem_task,  cem_cacc  = d["cem_task"],  d["cem_cacc"]
    gc_task,   gc_cacc   = d["gc_task"],   d["gc_cacc"]
    gcri_task, gcri_cacc = d["gcri_task"], d["gcri_cacc"]
    # CEM+RInt may not be in older cache — train if missing
    if "cemri_curves" in d:
        cemri_curves  = d["cemri_curves"]
        cemri_task, cemri_cacc = d["cemri_task"], d["cemri_cacc"]
        print("CEM+RInt loaded from cache.")
    else:
        print("CEM+RInt not in cache — training now...")
        cemri_curves_s, cemri_task_s, cemri_cacc_s = [], [], []
        for seed in range(N_SEEDS):
            print(f"  CEM+RInt seed={seed}", end="  ", flush=True)
            ta, ca, curve = train_cem_randint(seed)
            cemri_task_s.append(ta); cemri_cacc_s.append(ca); cemri_curves_s.append(curve)
            print(f"task={ta:.3f}  cacc={ca:.3f}  int_full={curve[-1]:.3f}")
        cemri_curves = np.array(cemri_curves_s)
        cemri_task, cemri_cacc = np.array(cemri_task_s), np.array(cemri_cacc_s)
        np.savez(CACHE_PATH,
                 cem_curves=cem_curves,   gc_curves=gc_curves,   gcri_curves=gcri_curves,
                 cem_task=cem_task,       cem_cacc=cem_cacc,
                 gc_task=gc_task,         gc_cacc=gc_cacc,
                 gcri_task=gcri_task,     gcri_cacc=gcri_cacc,
                 cemri_curves=cemri_curves, cemri_task=cemri_task, cemri_cacc=cemri_cacc)
        print(f"Cache updated → {CACHE_PATH}")
else:
    print(f"Training on Gaussian toy (K={K_TOTAL}, D={D_IN}, emb={EMB_SIZE})")
    print(f"N_SEEDS={N_SEEDS}, EPOCHS={EPOCHS}, lambda_adv={LAMBDA_ADV}, p_randint={P_RANDINT}\n")

    cem_curves_s,  gc_curves_s,  gcri_curves_s, cemri_curves_s = [], [], [], []
    cem_task_s,    gc_task_s,    gcri_task_s,   cemri_task_s   = [], [], [], []
    cem_cacc_s,    gc_cacc_s,    gcri_cacc_s,   cemri_cacc_s   = [], [], [], []

    for seed in range(N_SEEDS):
        print(f"--- seed={seed} ---")
        for label, fn, tc, cc, ic in [
            ("CEM",          train_cem,          cem_task_s,   cem_cacc_s,   cem_curves_s),
            ("CEM+RInt",     train_cem_randint,  cemri_task_s, cemri_cacc_s, cemri_curves_s),
            ("GC-CEM",       train_gccm,         gc_task_s,    gc_cacc_s,    gc_curves_s),
            ("GC-CEM+RInt",  train_gccm_randint, gcri_task_s,  gcri_cacc_s,  gcri_curves_s),
        ]:
            ta, ca, curve = fn(seed)
            tc.append(ta); cc.append(ca); ic.append(curve)
            print(f"  {label:15s}  task={ta:.3f}  cacc={ca:.3f}  int_full={curve[-1]:.3f}")

    cem_curves  = np.array(cem_curves_s)
    gc_curves   = np.array(gc_curves_s)
    gcri_curves = np.array(gcri_curves_s)
    cemri_curves = np.array(cemri_curves_s)
    cem_task,  cem_cacc   = np.array(cem_task_s),   np.array(cem_cacc_s)
    gc_task,   gc_cacc    = np.array(gc_task_s),    np.array(gc_cacc_s)
    gcri_task, gcri_cacc  = np.array(gcri_task_s),  np.array(gcri_cacc_s)
    cemri_task, cemri_cacc = np.array(cemri_task_s), np.array(cemri_cacc_s)

    np.savez(CACHE_PATH,
             cem_curves=cem_curves,   gc_curves=gc_curves,   gcri_curves=gcri_curves,
             cem_task=cem_task,       cem_cacc=cem_cacc,
             gc_task=gc_task,         gc_cacc=gc_cacc,
             gcri_task=gcri_task,     gcri_cacc=gcri_cacc,
             cemri_curves=cemri_curves, cemri_task=cemri_task, cemri_cacc=cemri_cacc)
    print(f"\nCache saved → {CACHE_PATH}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
xs = np.arange(K_TOTAL + 1)

C_CEMRI = "#f59b00"

all_models = [
    (cem_curves,   C_CEM,   "CEM"),
    (cemri_curves, C_CEMRI, "CEM+RInt"),
    (gc_curves,    C_GC,    "GC-CEM"),
    (gcri_curves,  C_GCRI,  "GC-CEM+RInt"),
]

fig, axes = plt.subplots(1, 4, figsize=(13, 3.0))
for ax, (curves, color, label) in zip(axes, all_models):
    mean = curves.mean(axis=0)
    se   = curves.std(axis=0) / np.sqrt(N_SEEDS)
    ax.plot(xs, mean, '-', color=color, lw=2, label=label)
    ax.fill_between(xs, mean - se, mean + se, alpha=0.2, color=color)
    ax.axhline(mean[0], color='gray', lw=0.8, ls=':', alpha=0.6)
    ax.set_xlabel("# interventions")
    ax.set_ylabel("Task accuracy")
    ax.set_ylim(0.5, 1.02)
    ax.set_title(label, fontsize=9)
    ax.set_xticks(range(0, K_TOTAL + 1, 4))

plt.suptitle(
    f"Intervention curves — K={K_TOTAL}, Gaussian toy, λ_adv={LAMBDA_ADV}, p_int={P_RANDINT}",
    fontsize=9, y=1.02
)
plt.tight_layout()
out = PLOT_DIR + "randint_gc_int_curves_K20.png"
plt.savefig(out); plt.close()
print(f"Saved → {out}")

# ── Overlay plot ──────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(5, 3.5))
for curves, color, label in all_models:
    mean = curves.mean(axis=0)
    se   = curves.std(axis=0) / np.sqrt(N_SEEDS)
    ax2.plot(xs, mean, '-', color=color, lw=2, label=label)
    ax2.fill_between(xs, mean - se, mean + se, alpha=0.15, color=color)

ax2.set_xlabel("# interventions")
ax2.set_ylabel("Task accuracy")
ax2.set_ylim(0.5, 1.02)
ax2.set_xticks(range(0, K_TOTAL + 1, 4))
ax2.legend(framealpha=0.85)
ax2.set_title(f"K={K_TOTAL} Gaussian toy — RandInt + GC-CEM", fontsize=9)
plt.tight_layout()
out2 = PLOT_DIR + "randint_gc_overlay_K20.png"
plt.savefig(out2); plt.close()
print(f"Saved → {out2}")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'Model':20s}  {'task acc':>10}  {'c_acc':>8}  {'int@0':>8}  {'int@full':>10}")
print("-" * 65)
for label, ta, ca, ic in [
    ("CEM",          cem_task,   cem_cacc,   cem_curves),
    ("CEM+RInt",     cemri_task, cemri_cacc, cemri_curves),
    ("GC-CEM",       gc_task,    gc_cacc,    gc_curves),
    ("GC-CEM+RInt",  gcri_task,  gcri_cacc,  gcri_curves),
]:
    print(f"  {label:18s}  {ta.mean():.3f}±{ta.std():.3f}  "
          f"{ca.mean():.3f}±{ca.std():.3f}  "
          f"{ic[:,0].mean():.3f}±{ic[:,0].std():.3f}  "
          f"{ic[:,-1].mean():.3f}±{ic[:,-1].std():.3f}")

print("\nDone.")
