"""
Same experiment as cvl_step1_quality_experiment.py but with CUB-scale parameters:
    K=112 concepts, C=200 classes, emb_size=16, N_TRAIN=8000.

Ridge-based CVL and ICVL only (KSG skipped entirely at this scale).

Key differences from the small version:
  - TRUE_CVL  ≈ 0.332  (large C → approaches the C→∞ formula)
  - TRUE_ICVL ≈ 0.003  (averaged over K*(K-1)=12432 pairs; only 112 adjacent pairs
                         contribute, the rest are ~0 — ICVL becomes very small)

Run with:
    conda run -n xai-leakage python experiments/evaluate_models/cvl_step1_quality_cub_experiment.py
"""
import os, sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import label_binarize
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)

SAVE_PATH = master + "results/results_cvl_step1_quality_cub.dict"

# ---------------------------------------------------------------------------
# CUB-scale parameters
# ---------------------------------------------------------------------------
K        = 112
C        = 200
EMB_SIZE = 16
N_TRAIN  = 8000
N_TEST   = 2000
ALPHA    = 1.0
SEEDS    = 3

SIGNAL_IC = 2.0
SIGMA     = 1.0

SIGNAL_C_VALS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def true_step1_r2(sig_c):
    return (0.25 * sig_c**2
            / (0.25 * sig_c**2 + 0.25 * SIGNAL_IC**2 + 1.0 + SIGMA**2))


# Analytical ground truth (see cvl_step1_quality_experiment.py for derivation)
_var_gamma = (C - 1) / C
_var_resid = 0.25 * SIGNAL_IC**2 + _var_gamma + SIGMA**2
TRUE_CVL   = _var_gamma / _var_resid
TRUE_ICVL  = (0.25 * SIGNAL_IC**2 / _var_resid) / (K - 1)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_data(N, sig_c, rng):
    c = rng.randint(0, 2, (N, K)).astype(float)
    y = rng.randint(0, C, N)

    gamma = [rng.randn(EMB_SIZE, C) for _ in range(K)]

    emb = np.zeros((N, K, EMB_SIZE))
    for k in range(K):
        k_next = (k + 1) % K
        emb[:, k, :] = (
            sig_c     * np.outer(c[:, k],      np.ones(EMB_SIZE))
            + SIGNAL_IC * np.outer(c[:, k_next], np.ones(EMB_SIZE))
            + gamma[k][:, y].T
            + SIGMA   * rng.randn(N, EMB_SIZE)
        )
    return emb, c, y


# ---------------------------------------------------------------------------
# Ridge-based CVL and ICVL
# ---------------------------------------------------------------------------
def estimate_cvl_icvl(emb_tr, emb_te, c_tr, c_te, y_tr, y_te):
    classes = np.arange(C)
    Y_tr = label_binarize(y_tr, classes=classes).astype(float)
    Y_te = label_binarize(y_te, classes=classes).astype(float)

    residuals_tr, residuals_te = {}, {}
    step1_r2s, cvl_vals = [], []

    for k in range(K):
        reg = Ridge(alpha=ALPHA)
        reg.fit(c_tr[:, k:k+1], emb_tr[:, k])
        R_tr = emb_tr[:, k] - reg.predict(c_tr[:, k:k+1])
        R_te = emb_te[:, k] - reg.predict(c_te[:, k:k+1])
        residuals_tr[k] = R_tr
        residuals_te[k] = R_te
        step1_r2s.append(r2_score(emb_te[:, k], reg.predict(c_te[:, k:k+1]),
                                  multioutput="variance_weighted"))

        reg2 = Ridge(alpha=ALPHA)
        reg2.fit(Y_tr, R_tr)
        cvl_vals.append(max(0.0, r2_score(R_te, reg2.predict(Y_te),
                                          multioutput="variance_weighted")))

    icvl_vals = []
    for k in range(K):
        for i in range(K):
            if i == k:
                continue
            reg = Ridge(alpha=ALPHA)
            reg.fit(c_tr[:, i:i+1], residuals_tr[k])
            icvl_vals.append(max(0.0,
                r2_score(residuals_te[k], reg.predict(c_te[:, i:i+1]),
                         multioutput="variance_weighted")))

    return float(np.mean(cvl_vals)), float(np.mean(icvl_vals)), float(np.mean(step1_r2s))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

print(f"K={K}  C={C}  EMB_SIZE={EMB_SIZE}  N_TRAIN={N_TRAIN}  seeds={SEEDS}")
print(f"SIGNAL_IC={SIGNAL_IC}  SIGMA={SIGMA}")
print(f"TRUE_CVL={TRUE_CVL:.4f}  TRUE_ICVL={TRUE_ICVL:.5f}")
print(f"Sweep: SIGNAL_C={SIGNAL_C_VALS}")
print(f"Analytical step-1 R²: {[round(true_step1_r2(s), 3) for s in SIGNAL_C_VALS]}")
print()

for sig_c in SIGNAL_C_VALS:
    key    = str(sig_c)
    cached = results.get(key, [])
    if len(cached) >= SEEDS:
        cvl  = np.mean([x["cvl"]      for x in cached])
        icvl = np.mean([x["icvl"]     for x in cached])
        r2   = np.mean([x["step1_r2"] for x in cached])
        print(f"SIGNAL_C={sig_c:5.2f}  R²={r2:.3f}  CVL={cvl:.4f}  ICVL={icvl:.5f}  [cached]")
        continue

    print(f"\nSIGNAL_C={sig_c:.2f}  (expected R²≈{true_step1_r2(sig_c):.3f})", flush=True)
    seed_res = []
    for seed in range(SEEDS):
        rng = np.random.RandomState(seed * 7919 + int(sig_c * 1000))
        print(f"  seed={seed}: generating...", end="", flush=True)
        emb, c, y = generate_data(N_TRAIN + N_TEST, sig_c, rng)
        print(" ridge...", end="", flush=True)
        cvl_v, icvl_v, s1r2 = estimate_cvl_icvl(
            emb[:N_TRAIN], emb[N_TRAIN:],
            c[:N_TRAIN],   c[N_TRAIN:],
            y[:N_TRAIN],   y[N_TRAIN:],
        )
        print(f"  R²={s1r2:.3f}  CVL={cvl_v:.4f}  ICVL={icvl_v:.5f}")
        seed_res.append({"cvl": cvl_v, "icvl": icvl_v, "step1_r2": s1r2})

    results[key] = seed_res
    joblib.dump(results, SAVE_PATH)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
print("\nGenerating plot ...")

present = [s for s in SIGNAL_C_VALS if results.get(str(s))]
x_r2    = np.array([np.mean([r["step1_r2"] for r in results[str(s)]]) for s in present])
order   = np.argsort(x_r2)
x_r2    = x_r2[order]


def _agg(metric):
    arr = np.array([[r[metric] for r in results[str(s)]] for s in present])
    return arr[order].mean(axis=1), arr[order].std(axis=1)


cvl_m,  cvl_s  = _agg("cvl")
icvl_m, icvl_s = _agg("icvl")

BLUE, RED = "#2166ac", "#d6604d"

fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax2 = ax1.twinx()

l1, = ax1.plot(x_r2, cvl_m, "o-", color=BLUE, linewidth=2, label="CVL (estimated)")
ax1.fill_between(x_r2, cvl_m - cvl_s, cvl_m + cvl_s, color=BLUE, alpha=0.18, linewidth=0)
lh1 = ax1.axhline(TRUE_CVL, color=BLUE, linestyle=":", linewidth=1.3,
                  label=f"true CVL = {TRUE_CVL:.3f}")

l2, = ax2.plot(x_r2, icvl_m, "s--", color=RED, linewidth=2, label="ICVL (estimated)")
ax2.fill_between(x_r2, icvl_m - icvl_s, icvl_m + icvl_s, color=RED, alpha=0.18, linewidth=0)
lh2 = ax2.axhline(TRUE_ICVL, color=RED, linestyle=":", linewidth=1.3,
                  label=f"true ICVL = {TRUE_ICVL:.4f}")

ax1.set_xlabel("Step-1 R²  (c_k explains embedding_k)", fontsize=11)
ax1.set_ylabel("CVL", color=BLUE, fontsize=11)
ax2.set_ylabel("ICVL", color=RED, fontsize=11)
ax1.tick_params(axis="y", labelcolor=BLUE)
ax2.tick_params(axis="y", labelcolor=RED)
ax1.set_xlim(0, 1)
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)

lines  = [l1, lh1, l2, lh2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, fontsize=9, loc="center right")

plt.title(
    f"CVL and ICVL vs step-1 explained variance — CUB scale\n"
    f"K={K}, C={C}, emb_size={EMB_SIZE}, SIGNAL_IC={SIGNAL_IC}, SIGMA={SIGMA}",
    fontsize=10,
)
plt.tight_layout()

for ext in (".pdf", ".png"):
    out = master + f"experiments/evaluate_models/cvl_step1_quality_cub{ext}"
    plt.savefig(out, bbox_inches="tight", dpi=150 if ext == ".png" else None)
    print(f"Saved → {out}")
