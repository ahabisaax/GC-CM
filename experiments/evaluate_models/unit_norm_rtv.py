"""
Compute unit-normalised RTV and RCV across all datasets, lambda_c values and folds.

For each concept k:
  1. Normalise embedding to unit variance (single scale per concept, fit on train)
  2. Step 1: Ridge(c_true_k -> emb_k_norm) -> residual_k_norm
     residual variance = 1 - step1_R2  (automatically, since emb_var=1)
  3. RTV_k = CVL_R2_k * resid_var_k  (= CVL_R2 * (1 - step1_R2))
  4. RCV_k = ICVL_R2_k * resid_var_k  (= ICVL_R2 * (1 - step1_R2))

Since R2 is scale-invariant, CVL_R2 and ICVL_R2 are unchanged by normalisation.
RTV and RCV are now in units of "fraction of unit embedding variance".
"""
import os, sys
import numpy as np
import joblib
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import label_binarize

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)

RIDGE_ALPHA = 1.0

DATASETS = [
    ("TabularToy", master + "results/embeddings_tabulartoy_all.joblib"),
    ("dSprites",   master + "results/embeddings_dsprites_all.joblib"),
    ("CUB",        master + "results/embeddings_cub_all.joblib"),
]

# fallback: use lam_c0.1 fold_1 from old CUB clean-split cache if new one absent
CUB_FALLBACK = master + "results/cub_embeddings_clean_split.joblib"


def compute_metrics(c_mix_tr, c_true_tr, y_tr, c_mix_te, c_true_te, y_te):
    N_tr, K, m = c_mix_tr.shape
    C = int(y_te.max()) + 1
    classes  = np.arange(C)
    Y_tr_oh  = label_binarize(y_tr, classes=classes).astype(float)
    Y_te_oh  = label_binarize(y_te, classes=classes).astype(float)

    step1_r2s, cvl_r2s, icvl_r2s, resid_vars = [], [], [], []

    for k in range(K):
        tr_k = c_mix_tr[:, k, :]
        te_k = c_mix_te[:, k, :]

        # Unit-normalise (single scale per concept, train stats)
        mu    = tr_k.mean()
        sigma = tr_k.std() + 1e-8
        tr_n  = (tr_k - mu) / sigma
        te_n  = (te_k - mu) / sigma

        # Step 1
        reg1 = Ridge(alpha=RIDGE_ALPHA)
        reg1.fit(c_true_tr[:, k:k+1], tr_n)
        pred1_tr = reg1.predict(c_true_tr[:, k:k+1])
        pred1_te = reg1.predict(c_true_te[:, k:k+1])
        s1_r2    = r2_score(te_n, pred1_te, multioutput="variance_weighted")
        step1_r2s.append(max(0.0, s1_r2))

        res_tr = tr_n - pred1_tr
        res_te = te_n - pred1_te
        resid_vars.append(float(np.var(res_te)))

        # CVL: Ridge(Y_onehot -> residual)
        reg2 = Ridge(alpha=RIDGE_ALPHA)
        reg2.fit(Y_tr_oh, res_tr)
        cvl_r2 = r2_score(res_te, reg2.predict(Y_te_oh), multioutput="variance_weighted")
        cvl_r2s.append(cvl_r2)

        # ICVL: mean over j != k of Ridge(c_true_j -> residual_k)
        icvl_jk = []
        for j in range(K):
            if j == k:
                continue
            reg3 = Ridge(alpha=RIDGE_ALPHA)
            reg3.fit(c_true_tr[:, j:j+1], res_tr)
            icvl_jk.append(r2_score(res_te, reg3.predict(c_true_te[:, j:j+1]),
                                     multioutput="variance_weighted"))
        icvl_r2s.append(float(np.mean(icvl_jk)))

    step1_r2  = float(np.mean(step1_r2s))
    resid_var = float(np.mean(resid_vars))   # ≈ 1 - step1_r2
    cvl_r2    = float(np.mean(cvl_r2s))
    icvl_r2   = float(np.mean(icvl_r2s))

    rtv = float(np.mean(np.maximum(0, np.array(cvl_r2s))  * np.array(resid_vars)))
    rcv = float(np.mean(np.maximum(0, np.array(icvl_r2s)) * np.array(resid_vars)))

    return dict(step1_r2=step1_r2, resid_var=resid_var,
                cvl_r2=cvl_r2, icvl_r2=icvl_r2, rtv=rtv, rcv=rcv)


# ── header ─────────────────────────────────────────────────────────────────
cols = ["Dataset", "Model", "lam_c", "step1_R2", "resid_var",
        "CVL_R2", "RTV", "ICVL_R2", "RCV"]
widths = [12, 10, 9, 9, 10, 9, 9, 9, 9]
hdr = "".join(f"{c:>{w}}" for c, w in zip(cols, widths))
print(hdr)
print("-" * len(hdr))

all_results = {}   # ds_name -> {model_lam -> avg_metrics}

for ds_name, path in DATASETS:
    if not os.path.exists(path):
        if ds_name == "CUB" and os.path.exists(CUB_FALLBACK):
            print(f"  {ds_name}: using fallback {CUB_FALLBACK}")
            raw_cache = joblib.load(CUB_FALLBACK)
            # old format: {"CEM": tuple, "GC-CEM": tuple}
            cache = {
                f"{m}_lam_c0.1_fold_1": raw_cache[m]
                for m in raw_cache
            }
        else:
            print(f"  {ds_name}: cache not found, skipping")
            continue
    else:
        cache = joblib.load(path)

    # Group by model + lam_c (strip fold)
    groups = defaultdict(list)
    for key, data in cache.items():
        model_lam = "_fold_".join(key.split("_fold_")[:-1])
        groups[model_lam].append(data)

    ds_results = {}
    for model_lam in sorted(groups.keys()):
        print(f"  Computing {ds_name} {model_lam} ({len(groups[model_lam])} folds)...",
              flush=True)
        fold_rs = [compute_metrics(*d) for d in groups[model_lam]]
        avg = {k: float(np.mean([f[k] for f in fold_rs])) for k in fold_rs[0]}
        ds_results[model_lam] = avg

        parts = model_lam.split("_lam_c")
        model = parts[0]
        lam   = "lam_c" + parts[1]
        vals  = [ds_name, model, lam,
                 avg["step1_r2"], avg["resid_var"],
                 avg["cvl_r2"], avg["rtv"],
                 avg["icvl_r2"], avg["rcv"]]
        row = "".join(f"{v:>{w}.4f}" if isinstance(v, float) else f"{v:>{w}}"
                      for v, w in zip(vals, widths))
        print(row)

    all_results[ds_name] = ds_results
    print()

joblib.dump(all_results, master + "results/unit_norm_rtv_rcv.dict")
print(f"\nSaved -> results/unit_norm_rtv_rcv.dict")
