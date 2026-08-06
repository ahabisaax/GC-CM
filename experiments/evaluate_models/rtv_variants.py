"""
RTV / RCV across three variants for all datasets × models × lambda_c.

Variant A  Ridge/Ridge : Step1=Ridge(c_true→emb),       Step2a=Ridge(Y→r),   Step2b=Ridge(c_true_j→r)
Variant B  Ridge/MLP   : Step1=Ridge(c_true→emb),       Step2a=MLP(r→Y),     Step2b=Ridge(c_true_j→r)
Variant C  MLP/MLP     : Step1=group_means(c_true→emb), Step2a=MLP(r→Y),     Step2b=group_means(c_true_j→r)

For binary concept labels, Ridge ≈ group_means (optimal linear = group means).
The main difference is Step2a direction: Ridge uses Y→r, MLP uses r→Y.
"""
import os, sys
import numpy as np
import joblib
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)

DATASETS = [
    ("TabularToy", master + "results/embeddings_tabulartoy_all.joblib"),
    ("dSprites",   master + "results/embeddings_dsprites_all.joblib"),
    ("CUB",        master + "results/embeddings_cub_all.joblib"),
]

RIDGE_ALPHA   = 1.0
MLP_HIDDEN    = 128
MLP_MAX_EPOCHS = 500
MLP_LR        = 1e-3
MLP_WD        = 1e-4
MLP_PATIENCE  = 20       # early-stopping patience (epochs without val improvement)
MLP_VAL_FRAC  = 0.2      # fraction of training data held out for early stopping
DEVICE        = "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def ridge_r2(X_tr, Y_tr, X_te, Y_te):
    reg = Ridge(alpha=RIDGE_ALPHA)
    reg.fit(X_tr, Y_tr)
    return float(r2_score(Y_te, reg.predict(X_te), multioutput="variance_weighted"))


def group_means_r2(x_tr, y_tr, x_te, y_te):
    """Binary x → continuous y. Group by x, predict y."""
    classes = np.unique(x_tr)
    y_hat = np.zeros_like(y_te, dtype=float)
    for c in classes:
        mask_tr = (x_tr == c)
        mask_te = (x_te == c)
        if mask_tr.sum() == 0:
            continue
        y_hat[mask_te] = y_tr[mask_tr].mean(axis=0)
    return float(r2_score(y_te, y_hat, multioutput="variance_weighted"))


class SmallMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def mlp_r2(X_tr, Y_tr, X_te, Y_te):
    """
    MLP regressor X→Y with early stopping on a held-out validation split.
    Returns R² evaluated on the held-out test set using the best-val-loss model.
    """
    N = X_tr.shape[0]
    n_val = max(1, int(N * MLP_VAL_FRAC))
    idx = np.random.permutation(N)
    tr_idx, val_idx = idx[n_val:], idx[:n_val]

    Xtr = torch.FloatTensor(X_tr[tr_idx]).to(DEVICE)
    Ytr = torch.FloatTensor(Y_tr[tr_idx]).to(DEVICE)
    Xvl = torch.FloatTensor(X_tr[val_idx]).to(DEVICE)
    Yvl = torch.FloatTensor(Y_tr[val_idx]).to(DEVICE)
    Xte = torch.FloatTensor(X_te).to(DEVICE)

    model     = SmallMLP(X_tr.shape[1], MLP_HIDDEN, Y_tr.shape[1]).to(DEVICE)
    opt       = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    best_val  = float("inf")
    best_state = None
    patience  = 0

    for epoch in range(MLP_MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        ((model(Xtr) - Ytr) ** 2).mean().backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = ((model(Xvl) - Yvl) ** 2).mean().item()

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1
            if patience >= MLP_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Y_hat = model(Xte).cpu().numpy()

    return float(r2_score(Y_te, Y_hat, multioutput="variance_weighted"))


# ---------------------------------------------------------------------------
# Per-entry metric computation
# ---------------------------------------------------------------------------

def compute_all_variants(c_mix_tr, c_true_tr, y_tr, c_mix_te, c_true_te, y_te, verbose_k=False):
    N_tr, K, m = c_mix_tr.shape
    C         = int(y_te.max()) + 1
    Y_tr_oh   = label_binarize(y_tr, classes=np.arange(C)).astype(np.float32)
    Y_te_oh   = label_binarize(y_te, classes=np.arange(C)).astype(np.float32)

    # per-concept accumulators for each variant
    s1_A, cv_A, ic_A, rv_A = [], [], [], []
    s1_B, cv_B, ic_B, rv_B = [], [], [], []
    s1_C, cv_C, ic_C, rv_C = [], [], [], []

    for k in range(K):
        tr_k = c_mix_tr[:, k, :]
        te_k = c_mix_te[:, k, :]

        mu    = tr_k.mean()
        sigma = tr_k.std() + 1e-8
        tr_n  = (tr_k - mu) / sigma
        te_n  = (te_k - mu) / sigma

        ck_tr = c_true_tr[:, k]
        ck_te = c_true_te[:, k]

        # ---------- Step 1 (variant A/B): Ridge(c_true_k → emb_k) ----------
        reg1 = Ridge(alpha=RIDGE_ALPHA).fit(ck_tr[:, None], tr_n)
        pred1_tr = reg1.predict(ck_tr[:, None])
        pred1_te = reg1.predict(ck_te[:, None])
        s1_r2_AB = max(0.0, r2_score(te_n, pred1_te, multioutput="variance_weighted"))
        r_AB_tr  = tr_n - pred1_tr
        r_AB_te  = te_n - pred1_te

        # ---------- Step 1 (variant C): group_means(c_true_k → emb_k) ----------
        s1_r2_C = max(0.0, group_means_r2(ck_tr, tr_n, ck_te, te_n))
        r_C_tr = tr_n.copy()
        r_C_te = te_n.copy()
        for c in np.unique(ck_tr):
            mask_tr = (ck_tr == c)
            mask_te = (ck_te == c)
            if mask_tr.sum() == 0:
                continue
            mean_c = tr_n[mask_tr].mean(axis=0)
            r_C_tr[mask_tr] -= mean_c
            r_C_te[mask_te] -= mean_c

        rv_AB = float(np.var(r_AB_te))
        rv_Ck = float(np.var(r_C_te))

        # ---------- Step 2a CVL ----------
        # A: Ridge(Y→r)
        cv_r2_A = ridge_r2(Y_tr_oh, r_AB_tr, Y_te_oh, r_AB_te)
        # B: MLP(r→Y)
        cv_r2_B = mlp_r2(r_AB_tr, Y_tr_oh, r_AB_te, Y_te_oh)
        # C: MLP(r→Y)
        cv_r2_C = mlp_r2(r_C_tr, Y_tr_oh, r_C_te, Y_te_oh)

        # ---------- Step 2b ICVL (Ridge for A/B, group_means for C) ----------
        icvl_AB, icvl_C = [], []
        for j in range(K):
            if j == k:
                continue
            cj_tr = c_true_tr[:, j]
            cj_te = c_true_te[:, j]
            icvl_AB.append(ridge_r2(cj_tr[:, None], r_AB_tr, cj_te[:, None], r_AB_te))
            icvl_C.append(group_means_r2(cj_tr, r_C_tr, cj_te, r_C_te))
        ic_r2_AB = float(np.mean(icvl_AB)) if icvl_AB else 0.0
        ic_r2_C  = float(np.mean(icvl_C))  if icvl_C  else 0.0

        s1_A.append(s1_r2_AB); cv_A.append(cv_r2_A); ic_A.append(ic_r2_AB); rv_A.append(rv_AB)
        s1_B.append(s1_r2_AB); cv_B.append(cv_r2_B); ic_B.append(ic_r2_AB); rv_B.append(rv_AB)
        s1_C.append(s1_r2_C);  cv_C.append(cv_r2_C); ic_C.append(ic_r2_C);  rv_C.append(rv_Ck)

        if verbose_k and (k + 1) % max(1, K // 4) == 0:
            print(f"    k={k+1}/{K}  s1={s1_r2_AB:.3f}  "
                  f"cvA={cv_r2_A:.3f}  cvB={cv_r2_B:.3f}  cvC={cv_r2_C:.3f}  "
                  f"rv={rv_AB:.3f}", flush=True)

    def agg(s1, cv, ic, rv):
        rv_arr = np.array(rv)
        rtv = float(np.mean(np.maximum(0, np.array(cv)) * rv_arr))
        rcv = float(np.mean(np.maximum(0, np.array(ic)) * rv_arr))
        return dict(
            step1_r2  = float(np.mean(s1)),
            resid_var = float(np.mean(rv)),
            cvl_r2    = float(np.mean(cv)),
            icvl_r2   = float(np.mean(ic)),
            rtv       = rtv,
            rcv       = rcv,
        )

    return {
        "A_Ridge_Ridge": agg(s1_A, cv_A, ic_A, rv_A),
        "B_Ridge_MLP":   agg(s1_B, cv_B, ic_B, rv_B),
        "C_MLP_MLP":     agg(s1_C, cv_C, ic_C, rv_C),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VAR_LABELS = {
    "A_Ridge_Ridge": "Ridge/Ridge",
    "B_Ridge_MLP":   "Ridge/MLP ",
    "C_MLP_MLP":     "MLP/MLP   ",
}

print(f"Device: {DEVICE}  MLP: hidden={MLP_HIDDEN} max_epochs={MLP_MAX_EPOCHS} patience={MLP_PATIENCE} val_frac={MLP_VAL_FRAC}")
print()

all_results = {}

for ds_name, path in DATASETS:
    if not os.path.exists(path):
        print(f"{ds_name}: cache not found, skipping")
        continue

    cache = joblib.load(path)

    groups = defaultdict(list)
    for key, data in cache.items():
        model_lam = "_fold_".join(key.split("_fold_")[:-1])
        groups[model_lam].append(data)

    ds_results = {}
    print(f"{'='*70}")
    print(f"  {ds_name}")
    print(f"{'='*70}")

    for model_lam in sorted(groups.keys()):
        parts  = model_lam.split("_lam_c")
        model  = parts[0]
        lam    = "lam_c" + parts[1]
        nfolds = len(groups[model_lam])
        print(f"\n  {model}  {lam}  ({nfolds} folds)", flush=True)

        fold_rs = [compute_all_variants(*d, verbose_k=True)
                   for d in groups[model_lam]]

        avgs = {}
        for var in fold_rs[0]:
            avgs[var] = {k: float(np.mean([f[var][k] for f in fold_rs]))
                         for k in fold_rs[0][var]}
        ds_results[model_lam] = avgs

        # Print comparison
        print(f"  {'Variant':<14}  {'s1_R2':>7}  {'resid_v':>7}  {'CVL_R2':>7}  {'RTV':>7}  {'ICVL_R2':>7}  {'RCV':>7}")
        print(f"  {'-'*62}")
        for var, label in VAR_LABELS.items():
            a = avgs[var]
            print(f"  {label}  {a['step1_r2']:7.4f}  {a['resid_var']:7.4f}  "
                  f"{a['cvl_r2']:7.4f}  {a['rtv']:7.4f}  "
                  f"{a['icvl_r2']:7.4f}  {a['rcv']:7.4f}")

    all_results[ds_name] = ds_results
    print()

joblib.dump(all_results, master + "results/rtv_variants.dict")
print(f"\nSaved → results/rtv_variants.dict")
