"""
RTL and RCL (sum and norm) over synthetic Gaussian toy experiments.

Sweeps:
  - RTL sweep: vary task-leakage coefficient, RCL=0
  - RCL sweep: vary inter-concept-leakage coefficient, RTL=0
  - Joint sweep: RTL=RCL varied together
  - Noise-dim sweep: fixed leakage, vary number of appended noise dims

Saves results dict and produces 4-panel plot.

Run from project root:
    python experiments/evaluate_models/run_rtl_gaussian.py
"""
import os, sys
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import Ridge
from sklearn.preprocessing import label_binarize

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)

# ── Config ────────────────────────────────────────────────────────────────────
RIDGE_ALPHA = 1.0
RNG_SEED    = 42
N_TRAIN     = 3000
N_TEST      = 1000
K           = 4       # concepts
M           = 16      # embedding dims per concept
C           = 4       # task classes (independent of concepts)
NOISE_SIGMA = 1.0

LEAKAGE_COEFS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
NOISE_DIMS    = [0, 4, 8, 16, 32, 48, 64]
FIXED_LEAKAGE = 2.0   # leakage level used in noise-dim sweep

OUT_DICT = master + "results/rtl_gaussian.dict"
OUT_PLOT = master + "results/rtl_gaussian.png"


# ── Data factory ──────────────────────────────────────────────────────────────

def make_data(leakage_rtl=0.0, leakage_rcl=0.0, n_noise_dims=0, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    N   = N_TRAIN + N_TEST

    c_true = rng.integers(0, 2, size=(N, K)).astype(np.float32)
    y      = rng.integers(0, C, size=N).astype(np.int32)   # independent of concepts
    y_oh   = label_binarize(y, classes=np.arange(C)).astype(np.float32)

    W = rng.standard_normal((K, M)).astype(np.float32)
    V = rng.standard_normal((K, C, M)).astype(np.float32)
    U = rng.standard_normal((K, M)).astype(np.float32)

    total_m = M + n_noise_dims
    c_mix   = np.zeros((N, K, total_m), dtype=np.float32)

    for k in range(K):
        j = (k + 1) % K
        emb  = c_true[:, k:k+1] * W[k:k+1, :]
        emb += leakage_rtl * (y_oh @ V[k])
        emb += leakage_rcl * c_true[:, j:j+1] * U[k:k+1, :]
        emb += rng.standard_normal((N, M)).astype(np.float32) * NOISE_SIGMA
        c_mix[:, k, :M] = emb
        if n_noise_dims > 0:
            c_mix[:, k, M:] = rng.standard_normal((N, n_noise_dims)).astype(np.float32)

    tr, te = slice(0, N_TRAIN), slice(N_TRAIN, N)
    return c_mix[tr], c_true[tr], y[tr], c_mix[te], c_true[te], y[te]


# ── Metric ────────────────────────────────────────────────────────────────────

def compute_rtl_rcl(c_mix_tr, c_true_tr, y_tr, c_mix_te, c_true_te, y_te):
    """Returns (rtl_sum, rcl_sum, rtl_norm, rcl_norm) averaged over concepts."""
    N_tr, K_, m = c_mix_tr.shape
    classes  = np.arange(int(y_te.max()) + 1)
    Y_tr_oh  = label_binarize(y_tr, classes=classes).astype(np.float32)
    Y_te_oh  = label_binarize(y_te, classes=classes).astype(np.float32)

    rtl_s, rcl_s, rtl_n, rcl_n = [], [], [], []

    for k in range(K_):
        tr_k = c_mix_tr[:, k, :]
        te_k = c_mix_te[:, k, :]

        mu    = tr_k.mean(axis=0, keepdims=True)
        sigma = tr_k.std(axis=0, keepdims=True) + 1e-8
        tr_n  = (tr_k - mu) / sigma
        te_n  = (te_k - mu) / sigma

        reg1 = Ridge(alpha=RIDGE_ALPHA).fit(c_true_tr[:, k:k+1], tr_n)
        r_tr = tr_n - reg1.predict(c_true_tr[:, k:k+1])
        r_te = te_n - reg1.predict(c_true_te[:, k:k+1])

        dim_var     = r_te.var(axis=0)
        total_resid = float(dim_var.sum()) + 1e-12
        ss_tot      = ((r_te - r_te.mean(axis=0)) ** 2).sum(axis=0)

        # RTL: Ridge(Y → r)
        reg2   = Ridge(alpha=RIDGE_ALPHA).fit(Y_tr_oh, r_tr)
        ss_res = ((r_te - reg2.predict(Y_te_oh)) ** 2).sum(axis=0)
        r2_y   = np.where(ss_tot > 1e-12, 1 - ss_res / ss_tot, 0.0)
        rtl_k  = float(np.maximum(0.0, r2_y * dim_var).sum())

        # RCL: Ridge(c_j → r) for j≠k
        rcl_j = []
        for j in range(K_):
            if j == k:
                continue
            reg3     = Ridge(alpha=RIDGE_ALPHA).fit(c_true_tr[:, j:j+1], r_tr)
            ss_res_j = ((r_te - reg3.predict(c_true_te[:, j:j+1])) ** 2).sum(axis=0)
            r2_j     = np.where(ss_tot > 1e-12, 1 - ss_res_j / ss_tot, 0.0)
            rcl_j.append(float(np.maximum(0.0, r2_j * dim_var).sum()))

        rcl_k = float(np.mean(rcl_j)) if rcl_j else 0.0

        rtl_s.append(rtl_k);          rcl_s.append(rcl_k)
        rtl_n.append(rtl_k / total_resid); rcl_n.append(rcl_k / total_resid)

    return (float(np.mean(rtl_s)), float(np.mean(rcl_s)),
            float(np.mean(rtl_n)), float(np.mean(rcl_n)))


# ── Run experiments ───────────────────────────────────────────────────────────

results = {}

print("RTL sweep (RCL=0)...")
rtl_sweep = {"coefs": LEAKAGE_COEFS, "rtl_sum": [], "rcl_sum": [], "rtl_norm": [], "rcl_norm": []}
for lam in LEAKAGE_COEFS:
    rs, cs, rn, cn = compute_rtl_rcl(*make_data(leakage_rtl=lam, leakage_rcl=0.0))
    rtl_sweep["rtl_sum"].append(rs); rtl_sweep["rcl_sum"].append(cs)
    rtl_sweep["rtl_norm"].append(rn); rtl_sweep["rcl_norm"].append(cn)
    print(f"  lam={lam:.2f}  RTL(sum)={rs:.4f}  RTL(norm)={rn:.4f}  RCL(sum)={cs:.4f}")
results["rtl_sweep"] = rtl_sweep

print("\nRCL sweep (RTL=0)...")
rcl_sweep = {"coefs": LEAKAGE_COEFS, "rtl_sum": [], "rcl_sum": [], "rtl_norm": [], "rcl_norm": []}
for lam in LEAKAGE_COEFS:
    rs, cs, rn, cn = compute_rtl_rcl(*make_data(leakage_rtl=0.0, leakage_rcl=lam))
    rcl_sweep["rtl_sum"].append(rs); rcl_sweep["rcl_sum"].append(cs)
    rcl_sweep["rtl_norm"].append(rn); rcl_sweep["rcl_norm"].append(cn)
    print(f"  lam={lam:.2f}  RCL(sum)={cs:.4f}  RCL(norm)={cn:.4f}  RTL(sum)={rs:.4f}")
results["rcl_sweep"] = rcl_sweep

print("\nJoint sweep (RTL=RCL)...")
joint_sweep = {"coefs": LEAKAGE_COEFS, "rtl_sum": [], "rcl_sum": [], "rtl_norm": [], "rcl_norm": []}
for lam in LEAKAGE_COEFS:
    rs, cs, rn, cn = compute_rtl_rcl(*make_data(leakage_rtl=lam, leakage_rcl=lam))
    joint_sweep["rtl_sum"].append(rs); joint_sweep["rcl_sum"].append(cs)
    joint_sweep["rtl_norm"].append(rn); joint_sweep["rcl_norm"].append(cn)
    print(f"  lam={lam:.2f}  RTL(sum)={rs:.4f}  RCL(sum)={cs:.4f}")
results["joint_sweep"] = joint_sweep

print(f"\nNoise-dim sweep (leakage={FIXED_LEAKAGE})...")
noise_sweep = {"noise_dims": NOISE_DIMS, "rtl_sum": [], "rcl_sum": [], "rtl_norm": [], "rcl_norm": []}
for nd in NOISE_DIMS:
    rs, cs, rn, cn = compute_rtl_rcl(*make_data(leakage_rtl=FIXED_LEAKAGE, leakage_rcl=FIXED_LEAKAGE, n_noise_dims=nd))
    noise_sweep["rtl_sum"].append(rs); noise_sweep["rcl_sum"].append(cs)
    noise_sweep["rtl_norm"].append(rn); noise_sweep["rcl_norm"].append(cn)
    print(f"  noise_dims={nd:3d}  total_m={M+nd:3d}  RTL(sum)={rs:.4f}  RTL(norm)={rn:.4f}")
results["noise_sweep"] = noise_sweep

joblib.dump(results, OUT_DICT)
print(f"\nSaved → {OUT_DICT}")


# ── Plotting ──────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

BLUE  = "#2979c0"
CORAL = "#e05c3a"
GREY  = "#888888"
LW    = 2.0
MS    = 6

def twin_plot(ax, x, y_sum, y_norm, xlabel, title, color_sum=BLUE, color_norm=CORAL):
    ax2 = ax.twinx()
    l1, = ax.plot(x, y_sum,  color=color_sum,  lw=LW, marker="o", ms=MS, label="sum")
    l2, = ax2.plot(x, y_norm, color=color_norm, lw=LW, marker="s", ms=MS, ls="--", label="norm [0,1]")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("RTL / RCL  (sum)", color=color_sum, fontsize=10)
    ax2.set_ylabel("RTL / RCL  (norm)", color=color_norm, fontsize=10)
    ax.tick_params(axis="y", colors=color_sum)
    ax2.tick_params(axis="y", colors=color_norm)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(handles=[l1, l2], loc="upper left", fontsize=9)
    return ax, ax2

# Panel 1 — RTL sweep
ax1 = fig.add_subplot(gs[0, 0])
twin_plot(ax1,
          LEAKAGE_COEFS,
          rtl_sweep["rtl_sum"],
          rtl_sweep["rtl_norm"],
          "Task-leakage coefficient λ",
          "RTL sweep  (RCL = 0)")

# Panel 2 — RCL sweep
ax2 = fig.add_subplot(gs[0, 1])
twin_plot(ax2,
          LEAKAGE_COEFS,
          rcl_sweep["rcl_sum"],
          rcl_sweep["rcl_norm"],
          "Inter-concept-leakage coefficient λ",
          "RCL sweep  (RTL = 0)",
          color_sum=CORAL, color_norm=BLUE)

# Panel 3 — Joint sweep (both RTL and RCL on same axes)
ax3 = fig.add_subplot(gs[1, 0])
ax3b = ax3.twinx()
l_rtl_s, = ax3.plot(LEAKAGE_COEFS,  joint_sweep["rtl_sum"],  color=BLUE,  lw=LW, marker="o", ms=MS, label="RTL sum")
l_rcl_s, = ax3.plot(LEAKAGE_COEFS,  joint_sweep["rcl_sum"],  color=CORAL, lw=LW, marker="^", ms=MS, label="RCL sum")
l_rtl_n, = ax3b.plot(LEAKAGE_COEFS, joint_sweep["rtl_norm"], color=BLUE,  lw=LW, marker="o", ms=MS, ls="--", label="RTL norm")
l_rcl_n, = ax3b.plot(LEAKAGE_COEFS, joint_sweep["rcl_norm"], color=CORAL, lw=LW, marker="^", ms=MS, ls="--", label="RCL norm")
ax3.set_xlabel("Leakage coefficient λ  (RTL = RCL)", fontsize=11)
ax3.set_ylabel("sum", fontsize=10)
ax3b.set_ylabel("norm [0,1]", fontsize=10)
ax3.set_title("Joint sweep  (RTL = RCL)", fontsize=12, fontweight="bold")
ax3.legend(handles=[l_rtl_s, l_rcl_s, l_rtl_n, l_rcl_n], loc="upper left", fontsize=8, ncol=2)

# Panel 4 — Noise-dim sweep
ax4 = fig.add_subplot(gs[1, 1])
ax4b = ax4.twinx()
total_ms = [M + nd for nd in NOISE_DIMS]
l1, = ax4.plot(total_ms,  noise_sweep["rtl_sum"],  color=BLUE,  lw=LW, marker="o", ms=MS, label="RTL sum")
l2, = ax4.plot(total_ms,  noise_sweep["rcl_sum"],  color=CORAL, lw=LW, marker="^", ms=MS, label="RCL sum")
l3, = ax4b.plot(total_ms, noise_sweep["rtl_norm"], color=BLUE,  lw=LW, marker="o", ms=MS, ls="--", label="RTL norm")
l4, = ax4b.plot(total_ms, noise_sweep["rcl_norm"], color=CORAL, lw=LW, marker="^", ms=MS, ls="--", label="RCL norm")
ax4.axvline(M, color=GREY, lw=1.2, ls=":", label=f"original m={M}")
ax4.set_xlabel(f"Total embedding dims  (signal={M}, rest=noise)", fontsize=11)
ax4.set_ylabel("sum  (should be flat)", fontsize=10)
ax4b.set_ylabel("norm  (expected to dilute)", fontsize=10)
ax4.set_title(f"Noise-dim invariance  (λ={FIXED_LEAKAGE})", fontsize=12, fontweight="bold")
ax4.legend(handles=[l1, l2, l3, l4], loc="upper right", fontsize=8, ncol=2)

fig.suptitle("RTL and RCL — synthetic Gaussian experiments\n"
             "sum = Σᵢ max(0, R²ᵢ × varᵢ)   |   norm = sum / Σᵢ varᵢ(residual)",
             fontsize=12, y=1.01)

plt.savefig(OUT_PLOT, dpi=150, bbox_inches="tight")
print(f"Plot saved → {OUT_PLOT}")
plt.show()
