"""
Robustness of RTL and RCL across K and L — paper generative model.

Mirrors cvl_robustness_paper_model.py (same generative model, same grid)
but uses RTL/RCL metrics instead of CVL/ICVL.

Grid: K in [6,10,20,50,112] x L in [2,8,32,100,200] x strength in [0,1,2,3]
RTL sweep: alpha (task leakage) in STRENGTHS, beta=0
RCL sweep: beta (inter-concept leakage) in STRENGTHS, alpha=0

Output: heatmaps saved to experiments/evaluate_models/

Run from project root:
    python experiments/evaluate_models/rtl_robustness_paper_model.py
"""
import os, sys
import numpy as np
from sklearn.preprocessing import label_binarize
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)

from xai_concept_leakage.metrics.leakage import compute_RTL_RCL

SAVE_PATH = master + "results/results_rtl_robustness_paper.dict"

# ---------------------------------------------------------------------------
# Parameters  (match cvl_robustness_paper_model.py)
# ---------------------------------------------------------------------------
N_TRAIN  = 2000
N_TEST   = 1000
M        = 16
SEEDS    = 20
N_REF    = 100_000

K_VALS    = [6, 10, 20, 50, 112]
L_VALS    = [2, 8, 32, 100, 200]
STRENGTHS = [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Data generation  (identical to cvl_robustness_paper_model.py)
# ---------------------------------------------------------------------------
def make_data(K, L, M, alpha, beta, seed):
    rng = np.random.RandomState(seed)

    w      = np.ones(K) if L == 2 else rng.randn(K)
    mu_pos = rng.randn(K, M)
    mu_neg = rng.randn(K, M)
    phi    = rng.randn(L, M)
    psi    = rng.randn(2, M)

    c_ref  = rng.randint(0, 2, (N_REF, K)).astype(float)
    s_ref  = c_ref @ w
    bounds = np.percentile(s_ref, np.linspace(0, 100, L + 1))

    def generate(n, rng_d):
        c = rng_d.randint(0, 2, (n, K)).astype(float)
        s = c @ w
        y = np.clip(np.digitize(s, bounds[1:-1]), 0, L - 1)

        emb = np.zeros((n, K, M))
        for k in range(K):
            emb[:, k] = (c[:, k:k+1] * mu_pos[k]
                         + (1 - c[:, k:k+1]) * mu_neg[k]
                         + rng_d.randn(n, M))

        if alpha > 0:
            emb += alpha * phi[y, np.newaxis, :]

        if beta > 0:
            psi_c0 = np.where(c[:, 0:1] == 1, psi[1], psi[0])
            emb[:, 1] += beta * psi_c0

        return emb, c, y

    rng_tr = np.random.RandomState(seed * 10_000 + 1)
    rng_te = np.random.RandomState(seed * 10_000 + 2)
    return generate(N_TRAIN, rng_tr), generate(N_TEST, rng_te)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

combos = (
    [(K, L, a, 0) for a in STRENGTHS for K in K_VALS for L in L_VALS] +
    [(K, L, 0, b) for b in STRENGTHS for K in K_VALS for L in L_VALS]
)
seen, combos_dedup = set(), []
for c in combos:
    if c not in seen:
        seen.add(c); combos_dedup.append(c)
total = len(combos_dedup)

print(f"Grid: K={K_VALS}  L={L_VALS}  strengths={STRENGTHS}")
print(f"RTL sweep: alpha in {STRENGTHS}, beta=0")
print(f"RCL sweep: beta in {STRENGTHS}, alpha=0")
print(f"Seeds={SEEDS}  N={N_TRAIN+N_TEST}  m={M}  Total unique cells={total}\n")

for done, (K, L, alpha, beta) in enumerate(combos_dedup, 1):
    key    = (K, L, alpha, beta)
    cached = results.get(key, [])

    if len(cached) >= SEEDS:
        rtl = np.mean([x["RTL_sum"] for x in cached])
        rcl = np.mean([x["RCL_sum"] for x in cached])
        print(f"[{done:3d}/{total}] K={K:3d} L={L:3d} a={alpha} b={beta} "
              f"[cached]  RTL_sum={rtl:.3f}  RCL_sum={rcl:.3f}")
        continue

    print(f"[{done:3d}/{total}] K={K:3d} L={L:3d} a={alpha} b={beta} ...",
          end="", flush=True)

    seed_res = list(cached)
    for seed in range(len(cached), SEEDS):
        (emb_tr, c_tr, y_tr), (emb_te, c_te, y_te) = make_data(K, L, M, alpha, beta, seed)
        r = compute_RTL_RCL(emb_tr, emb_te, c_tr, c_te, y_tr, y_te)
        seed_res.append(r)

    results[key] = seed_res
    joblib.dump(results, SAVE_PATH)

    rtl = np.mean([x["RTL_sum"] for x in seed_res])
    rcl = np.mean([x["RCL_sum"] for x in seed_res])
    print(f"  RTL_sum={rtl:.3f}  RCL_sum={rcl:.3f}")

print("\nGenerating plots ...")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def build_grid(metric, alpha, beta):
    g = np.full((len(K_VALS), len(L_VALS)), np.nan)
    for ki, K in enumerate(K_VALS):
        for li, L in enumerate(L_VALS):
            vals = [x[metric] for x in results.get((K, L, alpha, beta), []) if metric in x]
            if vals:
                g[ki, li] = np.mean(vals)
    return g


def heatmap(ax, grid, title, vmax):
    im = ax.imshow(grid, vmin=0, vmax=vmax, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(L_VALS))); ax.set_xticklabels(L_VALS, fontsize=8)
    ax.set_yticks(range(len(K_VALS))); ax.set_yticklabels(K_VALS, fontsize=8)
    ax.set_xlabel("Task classes (L)", fontsize=9)
    ax.set_title(title, fontsize=9, fontweight="bold")
    for ki in range(len(K_VALS)):
        for li in range(len(L_VALS)):
            v = grid[ki, li]
            if not np.isnan(v):
                ax.text(li, ki, f"{v:.2f}", ha="center", va="center", fontsize=7.5)
    return im


def save_plot(metric, strengths, alpha_fn, beta_fn, vmax, col_title_fn, cb_label, fname):
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8),
                             gridspec_kw={"wspace": 0.25})
    for col, s in enumerate(strengths):
        im = heatmap(axes[col], build_grid(metric, alpha_fn(s), beta_fn(s)),
                     col_title_fn(s), vmax)
        if col == 0:
            axes[col].set_ylabel("Concepts (K)", fontsize=9)

    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    cax  = fig.add_axes([0.25, -0.08, 0.5, 0.04])
    cb   = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="RdYlGn"),
                        cax=cax, orientation="horizontal")
    cb.set_label(cb_label, fontsize=9)
    cb.ax.tick_params(labelsize=8)

    for ext in [".pdf", ".png"]:
        out = master + f"experiments/evaluate_models/{fname}{ext}"
        plt.savefig(out, bbox_inches="tight", dpi=150 if ext == ".png" else None)
        print(f"Saved → {out}")
    plt.close()


vmax_rtl = max(np.nanmax(build_grid("RTL_sum", a, 0)) for a in STRENGTHS if a > 0) * 1.05
vmax_rcl = max(np.nanmax(build_grid("RCL_sum", 0, b)) for b in STRENGTHS if b > 0) * 1.05

save_plot(
    "RTL_sum", STRENGTHS,
    alpha_fn=lambda s: s, beta_fn=lambda s: 0,
    vmax=vmax_rtl,
    col_title_fn=lambda s: rf"RTL_sum  ($\alpha={s},\,\beta=0$)",
    cb_label="RTL_sum",
    fname="rtl_robustness_paper_heatmap_rtl",
)

save_plot(
    "RCL_sum", STRENGTHS,
    alpha_fn=lambda s: 0, beta_fn=lambda s: s,
    vmax=vmax_rcl,
    col_title_fn=lambda s: rf"RCL_sum  ($\alpha=0,\,\beta={s}$)",
    cb_label="RCL_sum",
    fname="rtl_robustness_paper_heatmap_rcl",
)

print("Done.")
