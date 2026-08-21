"""
OOD robustness experiment: Sequential CBM vs Joint CBM vs GC-CBM.

At test time, Gaussian noise is added to the concept predictions (c_hat) with
increasing sigma, simulating OOD concept representations (e.g. domain shift in
the concept predictor).  We measure task accuracy as a function of noise level.

Hypothesis: GC-CBM's task head is more robust because adversarial training
prevents it from relying on task-specific noise patterns in concept
representations, pushing c2y toward the true P(y | c) function.

Run from project root:
    python experiments/evaluate_models/validate_ood_cbm_robustness.py
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
    "font.family":    "DejaVu Sans",
    "font.size":      9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.75,
    "lines.linewidth": 1.8,
    "lines.markersize": 4,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":      True,
    "grid.alpha":     0.22,
    "grid.linewidth": 0.5,
    "figure.dpi":     300,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
})

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
K_TOTAL  = 15
D_IN     = 60          # 4 × K_TOTAL
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

NOISE_SIGMAS = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])

CACHE_PATH = f"results/cache/ood_cbm_robustness_K{K_TOTAL}.npz"
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

C_SEQ  = "#2ca02c"   # green  — Sequential CBM
C_JOINT = "#ff7f0e"  # orange — Joint CBM
C_GC   = "#d73027"   # red    — GC-CBM

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
# Gradient reversal
# ---------------------------------------------------------------------------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _mlp(in_dim, out_dim, hidden=32):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))


class SequentialCBM(nn.Module):
    """Two-phase: train x2c then freeze, train c2y on frozen predictions."""
    def __init__(self):
        super().__init__()
        self.x2c = _mlp(D_IN, K_TOTAL)
        self.c2y = _mlp(K_TOTAL, N_TASKS)

    def predict_c(self, x):
        return torch.sigmoid(self.x2c(x))

    def forward(self, x):
        c_hat = self.predict_c(x)
        return self.c2y(c_hat), c_hat


class JointCBM(nn.Module):
    """Joint training: concept loss + task loss simultaneously."""
    def __init__(self):
        super().__init__()
        self.x2c = _mlp(D_IN, K_TOTAL)
        self.c2y = _mlp(K_TOTAL, N_TASKS)

    def predict_c(self, x):
        return torch.sigmoid(self.x2c(x))

    def forward(self, x):
        c_hat = self.predict_c(x)
        return self.c2y(c_hat), c_hat


class GC_CBM(nn.Module):
    """Joint training + adversarial critic on residuals (c_hat - c_true)."""
    def __init__(self):
        super().__init__()
        self.x2c    = _mlp(D_IN, K_TOTAL)
        self.c2y    = _mlp(K_TOTAL, N_TASKS)
        self.critic = _mlp(K_TOTAL, N_TASKS)

    def predict_c(self, x):
        return torch.sigmoid(self.x2c(x))

    def forward(self, x):
        c_hat = self.predict_c(x)
        return self.c2y(c_hat), c_hat

    def residuals(self, c_hat, c_true):
        # difference between predicted and true binary concept values
        return c_hat - c_true


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------
def train_sequential(seed):
    torch.manual_seed(seed)
    x_all, c_all, y_all = generate_dataset(N_TOTAL, seed)
    x_tr, c_tr, y_tr = x_all[:N_TRAIN], c_all[:N_TRAIN], y_all[:N_TRAIN]
    x_te, c_te, y_te = x_all[N_TRAIN:], c_all[N_TRAIN:], y_all[N_TRAIN:]

    model = SequentialCBM()

    # Phase 1: train concept predictor only
    opt1  = torch.optim.Adam(model.x2c.parameters(), lr=LR)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, EPOCHS)
    dl = DataLoader(TensorDataset(x_tr, c_tr), batch_size=BATCH, shuffle=True)
    model.train()
    for _ in range(EPOCHS):
        for xb, cb in dl:
            opt1.zero_grad()
            F.binary_cross_entropy(torch.sigmoid(model.x2c(xb)), cb).backward()
            opt1.step()
        sched1.step()

    # Freeze x2c
    for p in model.x2c.parameters():
        p.requires_grad_(False)

    # Phase 2: train task head on frozen concept predictions
    opt2  = torch.optim.Adam(model.c2y.parameters(), lr=LR)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, EPOCHS)
    model.eval()
    with torch.no_grad():
        c_hat_tr_fixed = model.predict_c(x_tr)
    dl2 = DataLoader(TensorDataset(c_hat_tr_fixed, y_tr), batch_size=BATCH, shuffle=True)
    model.c2y.train()
    for _ in range(EPOCHS):
        for cb, yb in dl2:
            opt2.zero_grad()
            F.cross_entropy(model.c2y(cb), yb).backward()
            opt2.step()
        sched2.step()

    return model, x_te, c_te, y_te


def train_joint(seed, use_gc=False):
    torch.manual_seed(seed)
    x_all, c_all, y_all = generate_dataset(N_TOTAL, seed)
    x_tr, c_tr, y_tr = x_all[:N_TRAIN], c_all[:N_TRAIN], y_all[:N_TRAIN]
    x_te, c_te, y_te = x_all[N_TRAIN:], c_all[N_TRAIN:], y_all[N_TRAIN:]

    model = GC_CBM() if use_gc else JointCBM()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    dl    = DataLoader(TensorDataset(x_tr, c_tr, y_tr), batch_size=BATCH, shuffle=True)

    model.train()
    for _ in range(EPOCHS):
        for xb, cb, yb in dl:
            opt.zero_grad()
            y_hat, c_hat = model(xb)
            loss = F.cross_entropy(y_hat, yb) + LAMBDA_C * F.binary_cross_entropy(c_hat, cb)
            if use_gc:
                resid = grad_reverse(model.residuals(c_hat, cb), LAMBDA_ADV)
                loss  = loss + F.cross_entropy(model.critic(resid), yb)
            loss.backward()
            opt.step()
        sched.step()

    return model, x_te, c_te, y_te


def eval_ood(model, x_te, y_te, noise_sigmas):
    """Return task accuracy at each noise level."""
    model.eval()
    accs = []
    with torch.no_grad():
        c_hat = model.predict_c(x_te)
        for sigma in noise_sigmas:
            c_noisy = c_hat + torch.randn_like(c_hat) * sigma
            # no clamping — tests raw robustness of the task head
            y_pred  = model.c2y(c_noisy)
            acc     = (y_pred.argmax(-1) == y_te).float().mean().item()
            accs.append(acc)
    return np.array(accs)


# ---------------------------------------------------------------------------
# Main: train or load from cache
# ---------------------------------------------------------------------------
if os.path.exists(CACHE_PATH):
    print(f"Loading from cache: {CACHE_PATH}")
    _d = np.load(CACHE_PATH)
    seq_accs   = _d["seq_accs"]    # (N_SEEDS, len(NOISE_SIGMAS))
    joint_accs = _d["joint_accs"]
    gc_accs    = _d["gc_accs"]
else:
    print(f"Training on Gaussian toy (K_TOTAL={K_TOTAL}, D={D_IN})")
    seq_accs, joint_accs, gc_accs = [], [], []

    for seed in range(N_SEEDS):
        print(f"\n--- seed={seed} ---")

        print("  Sequential CBM ...", flush=True)
        m, x_te, c_te, y_te = train_sequential(seed)
        seq_accs.append(eval_ood(m, x_te, y_te, NOISE_SIGMAS))
        print(f"  clean acc={seq_accs[-1][0]:.3f}")

        print("  Joint CBM ...", flush=True)
        m, x_te, c_te, y_te = train_joint(seed, use_gc=False)
        joint_accs.append(eval_ood(m, x_te, y_te, NOISE_SIGMAS))
        print(f"  clean acc={joint_accs[-1][0]:.3f}")

        print("  GC-CBM ...", flush=True)
        m, x_te, c_te, y_te = train_joint(seed, use_gc=True)
        gc_accs.append(eval_ood(m, x_te, y_te, NOISE_SIGMAS))
        print(f"  clean acc={gc_accs[-1][0]:.3f}")

    seq_accs   = np.array(seq_accs)
    joint_accs = np.array(joint_accs)
    gc_accs    = np.array(gc_accs)

    np.savez(CACHE_PATH, seq_accs=seq_accs, joint_accs=joint_accs, gc_accs=gc_accs,
             noise_sigmas=NOISE_SIGMAS)
    print(f"\nCache saved → {CACHE_PATH}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
se = lambda arr: arr.std(0) / np.sqrt(N_SEEDS)

seq_mean   = seq_accs.mean(0);   seq_se   = se(seq_accs)
joint_mean = joint_accs.mean(0); joint_se = se(joint_accs)
gc_mean    = gc_accs.mean(0);    gc_se    = se(gc_accs)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

# Panel (a): accuracy vs noise sigma
ax = axes[0]
for mean, se_v, color, label in [
    (seq_mean,   seq_se,   C_SEQ,   "Sequential CBM"),
    (joint_mean, joint_se, C_JOINT, "Joint CBM"),
    (gc_mean,    gc_se,    C_GC,    "GC-CBM"),
]:
    ax.plot(NOISE_SIGMAS, mean, 'o-', color=color, label=label)
    ax.fill_between(NOISE_SIGMAS, mean - se_v, mean + se_v, alpha=0.18, color=color)

ax.set_xlabel("Concept noise $\\sigma$")
ax.set_ylabel("Task accuracy")
ax.set_title("(a) OOD robustness: task acc vs concept noise")
ax.legend()
ax.set_ylim(bottom=0.4)

# Panel (b): accuracy drop relative to sigma=0 (robustness curve)
ax2 = axes[1]
for mean, se_v, color, label in [
    (seq_mean,   seq_se,   C_SEQ,   "Sequential CBM"),
    (joint_mean, joint_se, C_JOINT, "Joint CBM"),
    (gc_mean,    gc_se,    C_GC,    "GC-CBM"),
]:
    drop    = mean[0] - mean
    drop_se = np.sqrt(se_v**2 + se_v[0]**2)
    ax2.plot(NOISE_SIGMAS, drop, 'o-', color=color, label=label)
    ax2.fill_between(NOISE_SIGMAS, drop - drop_se, drop + drop_se, alpha=0.18, color=color)

ax2.set_xlabel("Concept noise $\\sigma$")
ax2.set_ylabel("Accuracy drop from $\\sigma=0$")
ax2.set_title("(b) Accuracy degradation")
ax2.legend()
ax2.set_ylim(bottom=0)

plt.tight_layout()
os.makedirs("results/plots/cbm", exist_ok=True)
out = f"results/plots/cbm/ood_cbm_robustness_K{K_TOTAL}.png"
plt.savefig(out)
plt.close()
print(f"\nSaved → {out}")

# Summary table
print(f"\n{'sigma':>6}  {'Seq':>8}  {'Joint':>8}  {'GC-CBM':>8}  {'GC-Seq':>8}")
for i, s in enumerate(NOISE_SIGMAS):
    print(f"  {s:.2f}   {seq_mean[i]:.3f}    {joint_mean[i]:.3f}    {gc_mean[i]:.3f}"
          f"    {gc_mean[i]-seq_mean[i]:+.3f}")

print("\nDone.")
