"""
Experiment: CVL collapses as unmodelled "other" variance grows, while KSG CTL retains
sensitivity.

Hypothesis: CVL fails on real data (e.g. CUB, step-1 R²≈5%) not because leakage is
absent, but because the CEM embedding contains large amounts of variance that is neither
the concept label nor the task label — visual textures, object parts, etc. This "other"
variance swamps the CVL denominator.

Data model:

    e_k[n, j] = SIGNAL_C    * c_k[n]              (concept signal — fixed)
              + SIGNAL_IC   * c_{(k+1)%K}[n]       (inter-concept leakage — fixed)
              + SIGNAL_TASK * gamma_k[j, y[n]]      (task leakage — FIXED throughout)
              + SIGMA_OTHER * other_k[n, j]          (other variance — SWEPT)
              + SIGMA       * noise_j               (noise — fixed)

other_k[n, j] ~ N(0, 1), independent of c, y, and across concepts.

As SIGMA_OTHER increases:
  - Step-1 R² drops (concept explains less of the embedding)
  - True CVL = Var(task) / (Var(IC) + Var(task) + SIGMA_OTHER² + SIGMA²) → 0
  - Estimated CVL clips to 0 even sooner (negative R² clipped)
  - KSG CTL = MI(e_flat, y) − MI(c_true, y) also decreases, but more slowly and stays
    above the finite-sample noise floor longer

The gap between where CVL clips to 0 and where CTL detects no signal is the regime
in which CVL falsely reports zero leakage. At CUB (step-1 R²≈5%), we are deep in
this regime.

Run with:
    conda run -n xai-leakage python experiments/evaluate_models/cvl_other_variance_experiment.py
"""
import os, sys
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

SAVE_PATH = master + "results/results_cvl_other_variance.dict"

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------
K           = 6
C           = 2
EMB_SIZE    = 8
N_TRAIN     = 3000
N_TEST      = 1000
ALPHA       = 1.0
SEEDS       = 5
N_KSG_NEIGHBORS = 3
MAX_KSG_SAMPLES = 2000

SIGNAL_C    = 2.0   # fixed concept signal
SIGNAL_IC   = 1.0   # fixed inter-concept leakage
SIGNAL_TASK = 1.0   # fixed task leakage — THE THING WE WANT TO DETECT
SIGMA       = 0.5   # fixed noise

# Sweep: SIGMA_OTHER controls how much "other" variance is in the embedding
SIGMA_OTHER_VALS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]


def true_step1_r2(sigma_other):
    """Analytical R² of regressing c_k onto e_k (step 1 of CVL)."""
    var_signal = SIGNAL_C**2 * 0.25
    var_total  = (SIGNAL_C**2 * 0.25 + SIGNAL_IC**2 * 0.25
                  + _var_task + sigma_other**2 + SIGMA**2)
    return var_signal / var_total


def true_cvl(sigma_other):
    """Analytical true CVL for given SIGMA_OTHER."""
    denom = (SIGNAL_IC**2 * 0.25 + _var_task + sigma_other**2 + SIGMA**2)
    return _var_task / denom


def true_icvl(sigma_other):
    """Analytical true ICVL (per adjacent pair, averaged over K*(K-1) pairs)."""
    denom = (SIGNAL_IC**2 * 0.25 + _var_task + sigma_other**2 + SIGMA**2)
    return (SIGNAL_IC**2 * 0.25 / denom) / (K - 1)


# Expected Var(gamma[j, y]) for C classes
_var_task = (C - 1) / C * SIGNAL_TASK**2


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def generate_data(N, sigma_other, rng):
    c = rng.randint(0, 2, (N, K)).astype(float)
    y = rng.randint(0, C, N)

    gamma = [rng.randn(EMB_SIZE, C) * SIGNAL_TASK for _ in range(K)]

    emb = np.zeros((N, K, EMB_SIZE))
    for k in range(K):
        k_next = (k + 1) % K
        emb[:, k, :] = (
            SIGNAL_C    * np.outer(c[:, k],      np.ones(EMB_SIZE))
            + SIGNAL_IC   * np.outer(c[:, k_next], np.ones(EMB_SIZE))
            + gamma[k][:, y].T
            + sigma_other * rng.randn(N, EMB_SIZE)
            + SIGMA       * rng.randn(N, EMB_SIZE)
        )
    return emb, c, y


# ---------------------------------------------------------------------------
# Ridge-based CVL and ICVL
# ---------------------------------------------------------------------------
def estimate_cvl_icvl(emb_tr, emb_te, c_tr, c_te, y_tr, y_te):
    Y_tr = y_tr.reshape(-1, 1).astype(float)
    Y_te = y_te.reshape(-1, 1).astype(float)

    residuals_tr, residuals_te = {}, {}
    step1_r2s, cvl_vals, icvl_vals = [], [], []

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

    for k in range(K):
        for i in range(K):
            if i == k:
                continue
            reg = Ridge(alpha=ALPHA)
            reg.fit(c_tr[:, i:i+1], residuals_tr[k])
            icvl_vals.append(max(0.0,
                r2_score(residuals_te[k], reg.predict(c_te[:, i:i+1]),
                         multioutput="variance_weighted")))

    return (float(np.mean(cvl_vals)), float(np.mean(icvl_vals)),
            float(np.mean(step1_r2s)))


# ---------------------------------------------------------------------------
# KSG CTL
# ---------------------------------------------------------------------------
def estimate_ctl(emb, c, y, n_jobs):
    N = len(y)
    e_flat = emb.reshape(N, -1)

    if N > MAX_KSG_SAMPLES:
        idx    = np.random.choice(N, MAX_KSG_SAMPLES, replace=False)
        e_flat = e_flat[idx]
        c      = c[idx]
        y      = y[idx]

    mi_e_y = compute_mi_matrix_parallel(
        e_flat, d=y,
        n_neighbors=N_KSG_NEIGHBORS, normalise=True, flatten=False, n_jobs=n_jobs,
    )
    mi_c_y = compute_mi_matrix_parallel(
        c, d=y,
        n_neighbors=N_KSG_NEIGHBORS, normalise=True, flatten=False, n_jobs=n_jobs,
    )
    return float(np.maximum(0, np.mean(mi_e_y) - np.mean(mi_c_y)))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}
n_jobs  = os.cpu_count() or 1

print(f"K={K}  C={C}  EMB_SIZE={EMB_SIZE}  N_TRAIN={N_TRAIN}  seeds={SEEDS}")
print(f"SIGNAL_C={SIGNAL_C}  SIGNAL_IC={SIGNAL_IC}  SIGNAL_TASK={SIGNAL_TASK}  SIGMA={SIGMA}")
print(f"Sweep SIGMA_OTHER={SIGMA_OTHER_VALS}")
print(f"n_jobs={n_jobs}\n")
print(f"{'SIGMA_OTHER':>12}  {'step1_R²':>8}  {'TRUE_CVL':>9}  {'TRUE_ICVL':>10}")
for s in SIGMA_OTHER_VALS:
    print(f"{s:12.1f}  {true_step1_r2(s):8.3f}  {true_cvl(s):9.4f}  {true_icvl(s):10.5f}")
print()

for sigma_other in SIGMA_OTHER_VALS:
    key    = str(sigma_other)
    cached = results.get(key, [])
    if len(cached) >= SEEDS:
        cvl  = np.mean([x["cvl"]      for x in cached])
        icvl = np.mean([x["icvl"]     for x in cached])
        ctl  = np.mean([x["ctl"]      for x in cached])
        r2   = np.mean([x["step1_r2"] for x in cached])
        print(f"SIGMA_OTHER={sigma_other:5.1f}  R²={r2:.3f}  "
              f"CVL={cvl:.4f}  ICVL={icvl:.5f}  CTL={ctl:.4f}  [cached]")
        continue

    print(f"\nSIGMA_OTHER={sigma_other:.1f}  "
          f"(expected R²≈{true_step1_r2(sigma_other):.3f}  "
          f"TRUE_CVL={true_cvl(sigma_other):.4f})", flush=True)
    seed_res = []
    for seed in range(SEEDS):
        rng = np.random.RandomState(seed * 7919 + int(sigma_other * 100))
        emb, c, y = generate_data(N_TRAIN + N_TEST, sigma_other, rng)

        cvl_v, icvl_v, s1r2 = estimate_cvl_icvl(
            emb[:N_TRAIN], emb[N_TRAIN:],
            c[:N_TRAIN],   c[N_TRAIN:],
            y[:N_TRAIN],   y[N_TRAIN:],
        )
        ctl_v = estimate_ctl(emb[:N_TRAIN], c[:N_TRAIN], y[:N_TRAIN], n_jobs)

        seed_res.append({"cvl": cvl_v, "icvl": icvl_v, "ctl": ctl_v, "step1_r2": s1r2})
        print(f"  seed={seed}  R²={s1r2:.3f}  "
              f"CVL={cvl_v:.4f}  ICVL={icvl_v:.5f}  CTL={ctl_v:.4f}")

    results[key] = seed_res
    joblib.dump(results, SAVE_PATH)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
print("\nGenerating plot ...")

present = [s for s in SIGMA_OTHER_VALS if results.get(str(s))]
x_r2    = np.array([np.mean([r["step1_r2"] for r in results[str(s)]]) for s in present])
order   = np.argsort(x_r2)[::-1]   # descending R² (high → low, mirrors CUB direction)
x_r2    = x_r2[order]
x_so    = np.array(present)[order]  # corresponding sigma_other values


def _agg(metric):
    arr = np.array([[r[metric] for r in results[str(s)]] for s in present])
    return arr[order].mean(axis=1), arr[order].std(axis=1)


cvl_m,  cvl_s  = _agg("cvl")
icvl_m, icvl_s = _agg("icvl")
ctl_m,  ctl_s  = _agg("ctl")

true_cvl_arr  = np.array([true_cvl(s)  for s in x_so])
true_icvl_arr = np.array([true_icvl(s) for s in x_so])

BLUE, RED, GREEN = "#2166ac", "#d6604d", "#1a9641"

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# --- Panel 1: CVL ---
ax = axes[0]
ax.plot(x_r2, true_cvl_arr, color=BLUE, linestyle=":", linewidth=1.8,
        label="true CVL (analytical)")
ax.plot(x_r2, cvl_m, "o-", color=BLUE, linewidth=2, label="CVL (estimated, clipped ≥ 0)")
ax.fill_between(x_r2, cvl_m - cvl_s, cvl_m + cvl_s, color=BLUE, alpha=0.18, linewidth=0)
ax.axvline(0.05, color="grey", linestyle="--", linewidth=1.2, label="CUB step-1 R² ≈ 5%")
ax.set_xlabel("Step-1 R²  (lower = more 'other' variance)", fontsize=10)
ax.set_ylabel("CVL", fontsize=10)
ax.set_xlim(1.0, 0.0)   # right-to-left: high R² → low R²
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)
ax.set_title("CVL vs step-1 quality\n(task leakage held constant)", fontsize=10, fontweight="bold")

# --- Panel 2: ICVL ---
ax = axes[1]
ax.plot(x_r2, true_icvl_arr, color=RED, linestyle=":", linewidth=1.8,
        label="true ICVL (analytical)")
ax.plot(x_r2, icvl_m, "s-", color=RED, linewidth=2, label="ICVL (estimated, clipped ≥ 0)")
ax.fill_between(x_r2, icvl_m - icvl_s, icvl_m + icvl_s, color=RED, alpha=0.18, linewidth=0)
ax.axvline(0.05, color="grey", linestyle="--", linewidth=1.2, label="CUB step-1 R² ≈ 5%")
ax.set_xlabel("Step-1 R²  (lower = more 'other' variance)", fontsize=10)
ax.set_ylabel("ICVL", fontsize=10)
ax.set_xlim(1.0, 0.0)
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)
ax.set_title("ICVL vs step-1 quality\n(IC leakage held constant)", fontsize=10, fontweight="bold")

# --- Panel 3: CTL (KSG) ---
ax = axes[2]
ax.plot(x_r2, ctl_m, "D-", color=GREEN, linewidth=2, label="CTL (KSG estimated)")
ax.fill_between(x_r2, ctl_m - ctl_s, ctl_m + ctl_s, color=GREEN, alpha=0.18, linewidth=0)
ax.axvline(0.05, color="grey", linestyle="--", linewidth=1.2, label="CUB step-1 R² ≈ 5%")
ax.set_xlabel("Step-1 R²  (lower = more 'other' variance)", fontsize=10)
ax.set_ylabel("CTL (KSG)", fontsize=10)
ax.set_xlim(1.0, 0.0)
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)
ax.set_title("CTL (KSG) vs step-1 quality\n(task leakage held constant)", fontsize=10, fontweight="bold")

plt.suptitle(
    "Task leakage fixed — only 'other' embedding variance swept\n"
    f"K={K}, C={C}, emb_size={EMB_SIZE}, "
    f"SIGNAL_TASK={SIGNAL_TASK}, SIGNAL_IC={SIGNAL_IC}, SIGMA={SIGMA}",
    fontsize=10,
)
plt.tight_layout()

for ext in (".pdf", ".png"):
    out = master + f"experiments/evaluate_models/cvl_other_variance{ext}"
    plt.savefig(out, bbox_inches="tight", dpi=150 if ext == ".png" else None)
    print(f"Saved → {out}")
