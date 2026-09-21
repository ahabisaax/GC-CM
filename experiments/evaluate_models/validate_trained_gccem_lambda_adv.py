"""
Train tiny GC-CEMs on the Gaussian generative model with lambda_c fixed at 1,
sweeping the adversarial weight lambda_adv from 0 to 1, and measure RTL/RCL
(concept-vector level) and CTL/ICL (concept-probability level) at test time.

Why this experiment
-------------------
validate_lambda_c_rtl.py injects leakage analytically as alpha/lambda_c, so
its curves fall by construction. Here leakage is only ever produced or removed
by training pressure: at lambda_adv = 0 the model is an ordinary CEM, and
increasing lambda_adv strengthens the gradient-reversed critic that penalises
task information in the embeddings. If RTL and RCL fall monotonically, the
metrics are tracking the mechanism GC-CEM actually uses.

CEM is used rather than CBM deliberately: a CBM bottleneck is a K-vector of
probabilities and can only support CTL/ICL, whereas a CEM also exposes the
embedding geometry that RTL and RCL are defined on.

Generative model, encoder and CEM head are identical to
validate_trained_cem_lambda_c.py, so the two sweeps are comparable.

Metrics come from metrics.leakage.compute_RTL_RCL, which since 27179a2
normalises by the embedding dimension d, matching the paper's
RTL_k = (1/d) sum_m max(0, R^2_m) sigma^2_m. The earlier denominator (total
residual variance) is model-dependent and understated the adversarial effect.

Run from project root:
    python experiments/evaluate_models/validate_trained_gccem_lambda_adv.py
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

# ---------------------------------------------------------------------------
# Data / model parameters — mirrored from validate_trained_cem_lambda_c.py
# ---------------------------------------------------------------------------
N_TOTAL, N_TRAIN = 6000, 5000
K, K_TOTAL, D_IN, EMB_SIZE, N_TASKS = 6, 7, 24, 16, 2
GAMMA_EXTRA = 4.0
_rng42 = np.random.RandomState(42)
W_PROJ = torch.tensor(_rng42.randn(K_TOTAL, D_IN).astype(np.float32))

LAMBDA_C = 1.0
LAMBDA_ADV = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_SEEDS, EPOCHS, BATCH, LR = 5, 400, 256, 3e-3
ADV_LR = 3e-3
PLOT_DIR = "results/plots/cbm/"


def generate_dataset(n_samples, seed):
    rng = np.random.RandomState(seed)
    c_all = rng.randint(0, 2, (n_samples, K_TOTAL)).astype(np.float32)
    c_obs = c_all[:, :K]
    y = (c_all.sum(axis=1) >= (K_TOTAL / 2)).astype(np.int64)
    c_for_x = c_all.copy()
    c_for_x[:, K] *= GAMMA_EXTRA
    x = c_for_x @ W_PROJ.numpy() + rng.randn(n_samples, D_IN).astype(np.float32)
    return torch.tensor(x), torch.tensor(c_obs), torch.tensor(y)


class GradReverse(torch.autograd.Function):
    """Identity forward, negated and scaled gradient backward."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class TinyGCCEM(nn.Module):
    def __init__(self, input_dim, n_concepts, emb_size, n_tasks):
        super().__init__()
        self.x2c = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, n_concepts)
        )
        self.emb_pos = nn.Parameter(torch.randn(n_concepts, emb_size) * 0.1)
        self.emb_neg = nn.Parameter(torch.randn(n_concepts, emb_size) * 0.1)
        self.c2y = nn.Sequential(
            nn.Linear(n_concepts * emb_size, 32), nn.ReLU(), nn.Linear(32, n_tasks)
        )
        # Shared-critic setting, as in GCConceptEmbeddingModel: the critic has
        # the same shape as c2y and reads the same mixed embedding.
        self.critic = nn.Sequential(
            nn.Linear(n_concepts * emb_size, 32), nn.ReLU(), nn.Linear(32, n_tasks)
        )

    def forward(self, x, lam_adv=0.0):
        p = torch.sigmoid(self.x2c(x))
        p_exp = p.unsqueeze(-1)
        c_hat = p_exp * self.emb_pos + (1 - p_exp) * self.emb_neg
        c_flat = c_hat.reshape(x.size(0), -1)
        y_hat = self.c2y(c_flat)
        # One mixed tensor feeds BOTH branches (the 566261d invariant).
        y_adv = self.critic(GradReverse.apply(c_flat, lam_adv))
        return y_hat, y_adv, p, c_hat


def train(lam_adv, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, c, y = generate_dataset(N_TOTAL, seed)
    x_tr, c_tr, y_tr = x[:N_TRAIN], c[:N_TRAIN], y[:N_TRAIN]
    x_te, c_te, y_te = x[N_TRAIN:], c[N_TRAIN:], y[N_TRAIN:]
    m = TinyGCCEM(D_IN, K, EMB_SIZE, N_TASKS)
    opt = torch.optim.Adam(
        [
            {
                "params": [
                    q for n_, q in m.named_parameters() if not n_.startswith("critic")
                ]
            },
            {"params": m.critic.parameters(), "lr": ADV_LR},
        ],
        lr=LR,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    dl = DataLoader(TensorDataset(x_tr, c_tr, y_tr), batch_size=BATCH, shuffle=True)
    for _ in range(EPOCHS):
        for xb, cb, yb in dl:
            opt.zero_grad()
            y_hat, y_adv, p, _ = m(xb, lam_adv)
            loss = (
                F.cross_entropy(y_hat, yb)
                + LAMBDA_C * F.binary_cross_entropy(p, cb)
                + F.cross_entropy(y_adv, yb)
            )
            loss.backward()
            opt.step()
        sched.step()
    m.eval()
    with torch.no_grad():
        _, _, p_tr, e_tr = m(x_tr)
        y_hat_te, _, p_te, e_te = m(x_te)
        acc = (y_hat_te.argmax(1) == y_te).float().mean().item()
        cacc = ((p_te > 0.5).float() == c_te).float().mean().item()
    return (
        e_tr.numpy(),
        e_te.numpy(),
        c_tr.numpy(),
        c_te.numpy(),
        y_tr.numpy(),
        y_te.numpy(),
        acc,
        cacc,
    )


print(f"=== GC-CEM: lambda_c fixed at {LAMBDA_C}, sweeping lambda_adv ===")
print(f"    K={K} (+1 latent), d={EMB_SIZE}, {N_SEEDS} seeds, {EPOCHS} epochs\n")
res = {k: [] for k in ("rtl", "rcl", "acc", "cacc")}
for lam in LAMBDA_ADV:
    per = {k: [] for k in res}
    for s in range(N_SEEDS):
        e_tr, e_te, c_tr, c_te, y_tr, y_te, acc, cacc = train(lam, s)
        r = compute_RTL_RCL(e_tr, e_te, c_tr, c_te, y_tr, y_te, global_norm=True)
        per["rtl"].append(r["RTL_norm"])
        per["rcl"].append(r["RCL_norm"])
        per["acc"].append(acc)
        per["cacc"].append(cacc)
    for k in res:
        res[k].append((np.mean(per[k]), np.std(per[k], ddof=1)))
    print(
        f"  lambda_adv={lam:.1f}  RTL={res['rtl'][-1][0]:.4f}±{res['rtl'][-1][1]:.4f}"
        f"  RCL={res['rcl'][-1][0]:.4f}±{res['rcl'][-1][1]:.4f}"
        f"  task={res['acc'][-1][0]*100:.2f}%  concept={res['cacc'][-1][0]*100:.2f}%"
    )

os.makedirs(PLOT_DIR, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.1))
for ax, key, lab, col in (
    (axes[0], "rtl", "RTL", "#0072B2"),
    (axes[1], "rcl", "RCL", "#009E73"),
):
    mu = np.array([v[0] for v in res[key]])
    sd = np.array([v[1] for v in res[key]])
    ax.plot(LAMBDA_ADV, mu, "-o", color=col)
    ax.fill_between(LAMBDA_ADV, mu - sd, mu + sd, color=col, alpha=0.2)
    ax.set_xlabel(r"$\lambda_{adv}$")
    ax.set_ylabel(lab)
    ax.grid(alpha=0.25)
plt.tight_layout()
for ext in (".pdf", ".png"):
    plt.savefig(
        PLOT_DIR + "paper_trained_gccem_lambda_adv" + ext, bbox_inches="tight", dpi=150
    )
print("\nSaved -> " + PLOT_DIR + "paper_trained_gccem_lambda_adv.{pdf,png}")
