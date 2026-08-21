"""
Concept incompleteness experiment: train CEM and GC-CEM on a Gaussian
synthetic dataset where only k_obs of K_TOTAL concepts are supervised.

The task y = majority_vote(all K_TOTAL concepts), so as k_obs decreases
the supervised concepts explain less of y.  We track:
  - RTL/m      : task leakage in concept residuals
  - RCL/m      : inter-concept leakage in concept residuals
  - task_acc   : accuracy of y prediction
  - c_acc      : accuracy of supervised concept predictions
  - int_curve  : task accuracy as concepts are progressively intervened (0..k_obs)

Both CEM and GC-CEM are trained at fixed lambda_c=1.0; GC-CEM additionally
applies a gradient-reversal critic on concept residuals to suppress leakage.

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
K_TOTAL  = 15      # total concepts driving the task
D_IN     = 60      # input dimension (4 × K_TOTAL)
EMB_SIZE = 16      # embedding size per concept
N_TASKS  = 2       # binary task
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
class TinyCEM(nn.Module):
    def __init__(self, k_obs):
        super().__init__()
        self.k_obs = k_obs
        self.x2c   = nn.Sequential(
            nn.Linear(D_IN, 32), nn.ReLU(), nn.Linear(32, k_obs),
        )
        self.emb_pos = nn.Parameter(torch.randn(k_obs, EMB_SIZE) * 0.1)
        self.emb_neg = nn.Parameter(torch.randn(k_obs, EMB_SIZE) * 0.1)
        self.c2y     = nn.Sequential(
            nn.Linear(k_obs * EMB_SIZE, 32), nn.ReLU(), nn.Linear(32, N_TASKS),
        )

    def embed(self, x):
        p     = torch.sigmoid(self.x2c(x))
        p_exp = p.unsqueeze(-1)
        c_hat = p_exp * self.emb_pos + (1 - p_exp) * self.emb_neg
        return p, c_hat

    def forward(self, x):
        p, c_hat = self.embed(x)
        y_hat    = self.c2y(c_hat.reshape(x.size(0), -1))
        return y_hat, p, c_hat


class TinyGCCEM(TinyCEM):
    def __init__(self, k_obs):
        super().__init__(k_obs)
        self.critic = nn.Sequential(
            nn.Linear(k_obs * EMB_SIZE, 32), nn.ReLU(), nn.Linear(32, N_TASKS),
        )

    def residuals(self, c_hat, c_true):
        c_exp = c_true.unsqueeze(-1)
        proto = c_exp * self.emb_pos + (1 - c_exp) * self.emb_neg
        return c_hat - proto


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(ModelClass, k_obs, seed):
    torch.manual_seed(seed)
    x_all, c_all, y_all = generate_dataset(N_TOTAL, seed)

    x_tr = x_all[:N_TRAIN]; c_tr = c_all[:N_TRAIN, :k_obs]; y_tr = y_all[:N_TRAIN]
    x_te = x_all[N_TRAIN:]; c_te = c_all[N_TRAIN:, :k_obs]; y_te = y_all[N_TRAIN:]
    c_te_all = c_all[N_TRAIN:]

    dl = DataLoader(TensorDataset(x_tr, c_tr, y_tr), batch_size=BATCH, shuffle=True)

    model = ModelClass(k_obs)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    is_gc = isinstance(model, TinyGCCEM)

    model.train()
    for _ in range(EPOCHS):
        for xb, cb, yb in dl:
            opt.zero_grad()
            y_hat, p, c_hat = model(xb)
            loss = F.cross_entropy(y_hat, yb) + LAMBDA_C * F.binary_cross_entropy(p, cb)
            if is_gc:
                resid_r  = grad_reverse(
                    model.residuals(c_hat, cb).reshape(xb.size(0), -1), LAMBDA_ADV)
                loss = loss + F.cross_entropy(model.critic(resid_r), yb)
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        y_hat_te, p_te, c_hat_te = model(x_te)
        _, _, c_hat_tr = model(x_tr)

    task_acc    = (y_hat_te.argmax(-1) == y_te).float().mean().item()
    concept_acc = ((p_te > 0.5).float() == c_te).float().mean().item()

    # Intervention curve: replace 0, 1, ..., k_obs concept embeddings with clean prototypes
    int_curve = [task_acc]  # 0 interventions = model prediction
    with torch.no_grad():
        for n_int in range(1, k_obs + 1):
            c_hat_int = c_hat_te.clone()
            c_exp = c_te_all[:, :n_int].unsqueeze(-1)
            c_hat_int[:, :n_int] = c_exp * model.emb_pos[:n_int] + \
                                   (1 - c_exp) * model.emb_neg[:n_int]
            y_int = model.c2y(c_hat_int.reshape(len(x_te), -1))
            int_curve.append((y_int.argmax(-1) == y_te).float().mean().item())
    int_curve = np.array(int_curve)  # shape: (k_obs+1,)

    return (
        c_hat_tr.numpy(), c_hat_te.numpy(),
        c_tr.numpy(), c_te.numpy(),
        y_tr.numpy(), y_te.numpy(),
        task_acc, concept_acc, int_curve,
    )


# ---------------------------------------------------------------------------
# Sweep k_obs
# ---------------------------------------------------------------------------
CACHE_PATH = f"results/cache/concept_incompleteness_K{K_TOTAL}_v2.npz"
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

if os.path.exists(CACHE_PATH):
    print(f"Loading cached results from {CACHE_PATH}")
    _d = np.load(CACHE_PATH, allow_pickle=True)
    cem_rtl, cem_rtl_std = _d["cem_rtl"], _d["cem_rtl_std"]
    cem_rcl, cem_rcl_std = _d["cem_rcl"], _d["cem_rcl_std"]
    cem_task, cem_cacc   = _d["cem_task"], _d["cem_cacc"]
    gc_rtl,  gc_rtl_std  = _d["gc_rtl"],  _d["gc_rtl_std"]
    gc_rcl,  gc_rcl_std  = _d["gc_rcl"],  _d["gc_rcl_std"]
    gc_task, gc_cacc     = _d["gc_task"], _d["gc_cacc"]
    # int_curves: list of arrays of varying length
    cem_int_curves = list(_d["cem_int_curves"])
    gc_int_curves  = list(_d["gc_int_curves"])
else:
    print(f"Training on Gaussian toy (K_TOTAL={K_TOTAL}, D={D_IN}, m={EMB_SIZE})")
    cem_rtl, cem_rtl_std, cem_rcl, cem_rcl_std = [], [], [], []
    cem_task, cem_cacc, cem_int_curves = [], [], []
    gc_rtl,  gc_rtl_std,  gc_rcl,  gc_rcl_std  = [], [], [], []
    gc_task, gc_cacc, gc_int_curves = [], [], []

    for k_obs in K_OBS_VALUES:
        print(f"\n--- k_obs={k_obs}/{K_TOTAL} ---")
        for prefix, ModelClass, rl, rs, cl, cs, tl, cal, icl in [
            ("CEM",    TinyCEM,   cem_rtl, cem_rtl_std, cem_rcl, cem_rcl_std,
             cem_task, cem_cacc, cem_int_curves),
            ("GC-CEM", TinyGCCEM, gc_rtl,  gc_rtl_std,  gc_rcl,  gc_rcl_std,
             gc_task,  gc_cacc,  gc_int_curves),
        ]:
            rtl_v, rcl_v, ta_v, ca_v, ic_v = [], [], [], [], []
            for seed in range(N_SEEDS):
                print(f"  {prefix} seed={seed}", end="  ", flush=True)
                res = train_model(ModelClass, k_obs, seed)
                c_hat_tr, c_hat_te, c_tr, c_te, y_tr, y_te, ta, ca, ic = res
                r = compute_RTL_RCL(c_hat_tr, c_hat_te, c_tr, c_te, y_tr, y_te,
                                    global_norm=True)
                rtl_v.append(r["RTL_sum"] / EMB_SIZE)
                rcl_v.append(r["RCL_sum"] / EMB_SIZE)
                ta_v.append(ta); ca_v.append(ca); ic_v.append(ic)
                print(f"task={ta:.3f}  cacc={ca:.3f}  int_full={ic[-1]:.3f}"
                      f"  RTL={rtl_v[-1]:.4f}  RCL={rcl_v[-1]:.4f}")

            rl.append(np.mean(rtl_v)); rs.append(np.std(rtl_v))
            cl.append(np.mean(rcl_v)); cs.append(np.std(rcl_v))
            tl.append(np.mean(ta_v)); cal.append(np.mean(ca_v))
            icl.append(np.mean(ic_v, axis=0))  # mean curve across seeds

    def _a(l): return np.array(l)
    cem_rtl, cem_rtl_std = _a(cem_rtl), _a(cem_rtl_std)
    cem_rcl, cem_rcl_std = _a(cem_rcl), _a(cem_rcl_std)
    cem_task, cem_cacc   = _a(cem_task), _a(cem_cacc)
    gc_rtl,  gc_rtl_std  = _a(gc_rtl),  _a(gc_rtl_std)
    gc_rcl,  gc_rcl_std  = _a(gc_rcl),  _a(gc_rcl_std)
    gc_task, gc_cacc     = _a(gc_task), _a(gc_cacc)

    np.savez(CACHE_PATH,
             cem_rtl=cem_rtl, cem_rtl_std=cem_rtl_std,
             cem_rcl=cem_rcl, cem_rcl_std=cem_rcl_std,
             cem_task=cem_task, cem_cacc=cem_cacc,
             gc_rtl=gc_rtl,  gc_rtl_std=gc_rtl_std,
             gc_rcl=gc_rcl,  gc_rcl_std=gc_rcl_std,
             gc_task=gc_task, gc_cacc=gc_cacc,
             cem_int_curves=np.array(cem_int_curves, dtype=object),
             gc_int_curves=np.array(gc_int_curves,   dtype=object))
    print(f"\nCache saved → {CACHE_PATH}")


PLOT_DIR = "results/plots/cbm/"
os.makedirs(PLOT_DIR, exist_ok=True)
xs = np.array(K_OBS_VALUES)

# ---------------------------------------------------------------------------
# Figure 1: summary metrics vs k_obs
# ---------------------------------------------------------------------------
fig1, axes1 = plt.subplots(1, 4, figsize=(13, 2.8))

# For int_acc, extract final value of each curve
cem_int_final = np.array([c[-1] for c in cem_int_curves])
gc_int_final  = np.array([c[-1] for c in gc_int_curves])

panels = [
    (axes1[0], cem_rtl,       cem_rtl_std, gc_rtl,       gc_rtl_std, "RTL",                    "(a)"),
    (axes1[1], cem_rcl,       cem_rcl_std, gc_rcl,       gc_rcl_std, "RCL",                    "(b)"),
    (axes1[2], cem_task,      None,        gc_task,      None,       "Task accuracy",           "(c)"),
    (axes1[3], cem_int_final, None,        gc_int_final, None,       "Full intervention acc.",  "(d)"),
]

for ax, cem_m, cem_s, gc_m, gc_s, ylabel, letter in panels:
    ax.plot(xs, cem_m, color=C_CEM,  lw=1.9, marker="o", ms=4, label="CEM")
    ax.plot(xs, gc_m,  color=C_GCEM, lw=1.9, marker="s", ms=4, label="GC-CEM")
    if cem_s is not None:
        ax.fill_between(xs, cem_m - cem_s, cem_m + cem_s, color=C_CEM,  alpha=0.12)
        ax.fill_between(xs, gc_m  - gc_s,  gc_m  + gc_s,  color=C_GCEM, alpha=0.12)
    ax.set_xlabel("Observed concepts $k_{\\mathrm{obs}}$")
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs[::max(1, K_TOTAL // 8)])
    ax.set_xlim(xs[0] - 0.3, xs[-1] + 0.3)
    ax.set_ylim(bottom=0)
    ax.text(-0.18, 1.10, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

axes1[0].legend(loc="upper right", framealpha=0.9, edgecolor="0.78")
fig1.tight_layout(w_pad=1.2)
fig1.subplots_adjust(top=0.88)
out1 = PLOT_DIR + f"paper_concept_incompleteness_K{K_TOTAL}.pdf"
fig1.savefig(out1); fig1.savefig(out1.replace(".pdf", ".png"))
print(f"\nSaved → {out1}")
plt.close(fig1)

# ---------------------------------------------------------------------------
# Figure 2: intervention curves at selected k_obs values
# ---------------------------------------------------------------------------
# Pick ~4 representative k_obs values
n_show = min(4, K_TOTAL)
show_idx = np.round(np.linspace(0, K_TOTAL - 1, n_show)).astype(int)
show_kobs = [K_OBS_VALUES[i] for i in show_idx]

COLORS_K = plt.cm.viridis(np.linspace(0.15, 0.85, n_show))

fig2, axes2 = plt.subplots(1, 2, figsize=(8, 3.2))
for ax, model_curves, title, letter in [
    (axes2[0], cem_int_curves, "CEM",    "(a)"),
    (axes2[1], gc_int_curves,  "GC-CEM", "(b)"),
]:
    for j, (ki, k_obs) in enumerate(zip(show_idx, show_kobs)):
        curve = model_curves[ki]
        x_int = np.arange(len(curve))
        ax.plot(x_int, curve, color=COLORS_K[j], lw=1.8, marker="o", ms=3.5,
                label=f"$k_{{\\mathrm{{obs}}}}={k_obs}$")
    ax.set_xlabel("Number of concepts intervened")
    ax.set_ylabel("Task accuracy")
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_ylim(bottom=0, top=1.05)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="0.78", fontsize=7.5)
    ax.text(-0.15, 1.10, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top")

fig2.tight_layout(w_pad=1.8)
fig2.subplots_adjust(top=0.88)
out2 = PLOT_DIR + f"paper_concept_incompleteness_int_curves_K{K_TOTAL}.pdf"
fig2.savefig(out2); fig2.savefig(out2.replace(".pdf", ".png"))
print(f"Saved → {out2}")
plt.close(fig2)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print(f"\n{'k_obs':>6}  {'CEM RTL':>10}  {'GC RTL':>10}  "
      f"{'CEM RCL':>10}  {'GC RCL':>10}  "
      f"{'CEM task':>10}  {'GC task':>10}  "
      f"{'CEM int':>10}  {'GC int':>10}")
print("-" * 96)
for i, k in enumerate(K_OBS_VALUES):
    print(f"{k:>6}  {cem_rtl[i]:>10.4f}  {gc_rtl[i]:>10.4f}  "
          f"{cem_rcl[i]:>10.4f}  {gc_rcl[i]:>10.4f}  "
          f"{cem_task[i]:>10.4f}  {gc_task[i]:>10.4f}  "
          f"{cem_int_final[i]:>10.4f}  {gc_int_final[i]:>10.4f}")
print("\nDone.")
