"""
Unit-normalised RTV and RCV using MLP for Step 2a (RTV).

Step 1: group conditional means (binary c_true -> embedding; optimal for binary predictor)
Step 2a (RTV): MLP(r_k -> Y_onehot), R^2 x v_k
Step 2b (RCV): group conditional means given c_true_j (binary -> optimal)

For Steps 1 and 2b, MLP reduces to group means for binary predictors, so we use
the closed-form directly.
"""
import os, sys
import numpy as np
import joblib
from collections import defaultdict
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
CUB_FALLBACK = master + "results/cub_embeddings_clean_split.joblib"

# MLP hyperparameters
MLP_HIDDEN  = 256
MLP_EPOCHS  = 300
MLP_LR      = 1e-3
MLP_WD      = 1e-4
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def group_means_r2(x_tr, y_tr, x_te, y_te):
    """
    R^2 for predicting y (N x d) from binary x (N,) via group conditional means.
    Groups are defined by x (binary label); y is the continuous target.
    Fit means on train, evaluate on test.
    """
    classes = np.unique(x_tr)          # group by x, not y
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
    Fit MLP regressor (X -> Y), return R^2 on test.
    X shape: (N, in_dim); Y shape: (N, out_dim).
    """
    X_tr_t = torch.FloatTensor(X_tr).to(DEVICE)
    Y_tr_t = torch.FloatTensor(Y_tr).to(DEVICE)
    X_te_t = torch.FloatTensor(X_te).to(DEVICE)

    model = SmallMLP(X_tr.shape[1], MLP_HIDDEN, Y_tr.shape[1]).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=MLP_WD)

    for _ in range(MLP_EPOCHS):
        model.train()
        opt.zero_grad()
        loss = ((model(X_tr_t) - Y_tr_t) ** 2).mean()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        Y_hat = model(X_te_t).cpu().numpy()

    return float(r2_score(Y_te, Y_hat, multioutput="variance_weighted"))


# ---------------------------------------------------------------------------
# Per-entry computation
# ---------------------------------------------------------------------------

def compute_metrics(c_mix_tr, c_true_tr, y_tr, c_mix_te, c_true_te, y_te):
    N_tr, K, m = c_mix_tr.shape
    C = int(y_te.max()) + 1
    classes  = np.arange(C)
    Y_tr_oh  = label_binarize(y_tr, classes=classes).astype(np.float32)
    Y_te_oh  = label_binarize(y_te, classes=classes).astype(np.float32)

    step1_r2s, cvl_r2s, icvl_r2s, resid_vars = [], [], [], []

    for k in range(K):
        tr_k = c_mix_tr[:, k, :]
        te_k = c_mix_te[:, k, :]

        # Unit-normalise
        mu    = tr_k.mean()
        sigma = tr_k.std() + 1e-8
        tr_n  = (tr_k - mu) / sigma
        te_n  = (te_k - mu) / sigma

        # Step 1: group conditional means (binary c_true_k -> embedding)
        s1_r2   = group_means_r2(c_true_tr[:, k], tr_n, c_true_te[:, k], te_n)
        step1_r2s.append(max(0.0, s1_r2))

        # Residual
        for c in np.unique(c_true_tr[:, k]):
            mask_tr = c_true_tr[:, k] == c
            mask_te = c_true_te[:, k] == c
            if mask_tr.sum() == 0:
                continue
            mean_c = tr_n[mask_tr].mean(axis=0)
            tr_n[mask_tr] -= mean_c
            te_n[mask_te] -= mean_c

        resid_vars.append(float(np.var(te_n)))

        # Step 2a (RTV): MLP(r_k -> Y_onehot)
        cvl_r2 = mlp_r2(tr_n, Y_tr_oh, te_n, Y_te_oh)
        cvl_r2s.append(cvl_r2)

        # Step 2b (RCV): group means of r_k given c_true_j, for each j != k
        icvl_jk = []
        for j in range(K):
            if j == k:
                continue
            r2_jk = group_means_r2(c_true_tr[:, j], tr_n, c_true_te[:, j], te_n)
            icvl_jk.append(r2_jk)
        icvl_r2s.append(float(np.mean(icvl_jk)))

        if (k + 1) % max(1, K // 4) == 0:
            print(f"    concept {k+1}/{K}  step1={step1_r2s[-1]:.3f}  "
                  f"cvl={cvl_r2s[-1]:.4f}  resid_var={resid_vars[-1]:.4f}", flush=True)

    step1_r2  = float(np.mean(step1_r2s))
    resid_var = float(np.mean(resid_vars))
    cvl_r2    = float(np.mean(cvl_r2s))
    icvl_r2   = float(np.mean(icvl_r2s))
    rtv = float(np.mean(np.maximum(0, np.array(cvl_r2s))  * np.array(resid_vars)))
    rcv = float(np.mean(np.maximum(0, np.array(icvl_r2s)) * np.array(resid_vars)))

    return dict(step1_r2=step1_r2, resid_var=resid_var,
                cvl_r2=cvl_r2, icvl_r2=icvl_r2, rtv=rtv, rcv=rcv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

cols   = ["Dataset", "Model", "lam_c", "step1_R2", "resid_var",
          "CVL_R2", "RTV", "ICVL_R2", "RCV"]
widths = [12, 10, 9, 9, 10, 9, 9, 9, 9]
hdr    = "".join(f"{c:>{w}}" for c, w in zip(cols, widths))
print(f"Device: {DEVICE}  MLP: hidden={MLP_HIDDEN} epochs={MLP_EPOCHS}")
print(hdr)
print("-" * len(hdr))

all_results = {}

for ds_name, path in DATASETS:
    if not os.path.exists(path):
        if ds_name == "CUB" and os.path.exists(CUB_FALLBACK):
            print(f"  {ds_name}: using fallback")
            raw = joblib.load(CUB_FALLBACK)
            cache = {f"{m}_lam_c0.1_fold_1": raw[m] for m in raw}
        else:
            print(f"  {ds_name}: not found, skipping")
            continue
    else:
        cache = joblib.load(path)

    groups = defaultdict(list)
    for key, data in cache.items():
        model_lam = "_fold_".join(key.split("_fold_")[:-1])
        groups[model_lam].append(data)

    ds_results = {}
    for model_lam in sorted(groups.keys()):
        print(f"\n  {ds_name} {model_lam} ({len(groups[model_lam])} folds)", flush=True)
        fold_rs = [compute_metrics(*d) for d in groups[model_lam]]
        avg = {k: float(np.mean([f[k] for f in fold_rs])) for k in fold_rs[0]}
        ds_results[model_lam] = avg

        parts = model_lam.split("_lam_c")
        model, lam = parts[0], "lam_c" + parts[1]
        vals = [ds_name, model, lam,
                avg["step1_r2"], avg["resid_var"],
                avg["cvl_r2"], avg["rtv"], avg["icvl_r2"], avg["rcv"]]
        row = "".join(f"{v:>{w}.4f}" if isinstance(v, float) else f"{v:>{w}}"
                      for v, w in zip(vals, widths))
        print(row)

    all_results[ds_name] = ds_results
    print()

joblib.dump(all_results, master + "results/mlp_rtv_rcv.dict")
print(f"\nSaved -> results/mlp_rtv_rcv.dict")
