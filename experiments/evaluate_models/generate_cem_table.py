"""
LaTeX table: CEM vs GC-CEM across TabularToy / dSprites / CUB-200.
Run from project root: python experiments/evaluate_models/generate_cem_table.py
"""
import glob, joblib, numpy as np, os, sys
sys.path.insert(0, os.getcwd())

R   = "results/"
lp  = lambda l: "1" if l == 1.0 else str(l)
KEYS = ["test_acc_y", "test_acc_c", "test_ctl_average", "test_icl_average"]
LAMS = [0.1, 0.5, 1.0]
LAM_KEYS = ["lam_c0.1", "lam_c0.5", "lam_c1"]


def from_splits(folder, mdir):
    splits = sorted(glob.glob(f"{folder}/{mdir}/*_split_*_results.joblib"))
    if not splits:
        return None
    out = {k: [] for k in KEYS}
    for s in splits:
        d = joblib.load(s)
        for k in KEYS:
            out[k].append(d.get(k, np.nan))
    return {k: np.array(v) for k, v in out.items()}


def cell(arr, pct=False):
    arr = np.asarray(arr, float)
    arr = arr[~np.isnan(arr)]
    if not len(arr):
        return "--"
    m, s = np.mean(arr), np.std(arr)
    return f"{m*100:.1f} \\pm {s*100:.1f}" if pct else f"{m:.3f} \\pm {s:.3f}"


def bcell(arr, best, pct=False):
    c = cell(arr, pct)
    if c == "--" or best is None:
        return c
    arr = np.asarray(arr, float)
    arr = arr[~np.isnan(arr)]
    if len(arr) and abs(arr.mean() - best) < 1e-9:
        return r"\textbf{" + c + r"}"
    return c


# ---- collect rows: (dataset, model, lam, task, cacc, ctl, icl) ----
rows = []

# TabularToy — from suite dict
tt = joblib.load(R + "results_tabulartoy_acem_suite.dict")
for lam_c, lk in zip(LAMS, LAM_KEYS):
    for model, key in [("Joint CEM", "cem"), ("GC-CEM (ours)", "acem")]:
        folds = tt[key][lk]
        task = np.array([v["task_acc"] for v in folds.values()])
        cacc = np.array([v["c_acc"] for v in folds.values()])
        ctl  = np.array([np.mean(v["ctl_mean"]) for v in folds.values()])
        icl  = np.array([v["icl_mean"] for v in folds.values()])
        rows.append(("TabularToy", model, lam_c, task, cacc, ctl, icl))

# dSprites — from split joblibs
for lam in LAMS:
    for model, folder, mdir in [
        ("Joint CEM",     R + "dsprites_cem",
         "CEM_adam_lr1e-03_bs64_lam_c" + lp(lam)),
        ("GC-CEM (ours)", R + "dsprites_acem_shared_critic",
         "CRCEM_adam_lr5e-05_bs64_lam1_none_lam_c" + lp(lam) + "_shared_critic"),
    ]:
        d = from_splits(folder, mdir)
        if d:
            rows.append(("dSprites", model, lam,
                         d["test_acc_y"], d["test_acc_c"],
                         d["test_ctl_average"], d["test_icl_average"]))

# CUB-200 — from split joblibs
for lam in LAMS:
    for model, folder, mdir in [
        ("Joint CEM",     R + "cub_cem",
         "CEM_adam_lr1e-03_bs256_lam_c" + lp(lam)),
        ("GC-CEM (ours)", R + "cub_cem",
         "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c" + lp(lam) + "_shared_critic"),
    ]:
        d = from_splits(folder, mdir)
        if d:
            rows.append(("CUB-200", model, lam,
                         d["test_acc_y"], d["test_acc_c"],
                         d["test_ctl_average"], d["test_icl_average"]))

# ---- build LaTeX ----
lines = []
lines.append(r"\begin{table*}[t]")
lines.append(r"\centering")
lines.append(
    r"\caption{Joint CEM vs GC-CEM across datasets and $\lambda_c$. "
    r"Results are mean $\pm$ std over 5 seeds. "
    r"Best result per (dataset, $\lambda_c$) is \textbf{bolded}. "
    r"Task and concept accuracy are in \%.}"
)
lines.append(r"\label{tab:cem_gccpm}")
lines.append(r"\resizebox{\textwidth}{!}{%")
lines.append(r"\begin{tabular}{llc cccc}")
lines.append(r"\toprule")
lines.append(
    r"\textbf{Dataset} & \textbf{Model} & $\lambda_c$ "
    r"& Task Acc $\uparrow$ & Concept Acc $\uparrow$ "
    r"& CTL $\downarrow$ & ICL $\downarrow$ \\"
)
lines.append(r"\midrule")

DATASETS = ["TabularToy", "dSprites", "CUB-200"]
for di, ds in enumerate(DATASETS):
    ds_rows = [r for r in rows if r[0] == ds]
    n_total = len(ds_rows)
    first_ds = True
    for li, lam in enumerate(LAMS):
        lam_rows = [r for r in ds_rows if r[2] == lam]
        def _mean(arr):
            arr = np.asarray(arr, float); arr = arr[~np.isnan(arr)]
            return arr.mean() if len(arr) else None
        best_task = max((_mean(r[3]) for r in lam_rows if _mean(r[3]) is not None), default=None)
        best_cacc = max((_mean(r[4]) for r in lam_rows if _mean(r[4]) is not None), default=None)
        best_ctl  = min((_mean(r[5]) for r in lam_rows if _mean(r[5]) is not None), default=None)
        best_icl  = min((_mean(r[6]) for r in lam_rows if _mean(r[6]) is not None), default=None)
        for ri, row in enumerate(lam_rows):
            _, model, _, task, cacc, ctl, icl = row
            ds_col  = (r"\multirow{" + str(n_total) + r"}{*}{\textbf{" + ds + r"}}"
                       if (first_ds and ri == 0) else "")
            lam_col = (r"\multirow{2}{*}{$" + str(lam) + r"$}" if ri == 0 else "")
            if first_ds and ri == 0:
                first_ds = False
            parts = [
                ds_col, model, lam_col,
                bcell(task, best_task, True),
                bcell(cacc, best_cacc, True),
                bcell(ctl,  best_ctl,  False),
                bcell(icl,  best_icl,  False),
            ]
            lines.append(" & ".join(parts) + r" \\")
        if li < len(LAMS) - 1:
            lines.append(r"\cmidrule(lr){2-7}")
    if di < len(DATASETS) - 1:
        lines.append(r"\midrule")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}%")
lines.append(r"}")
lines.append(r"\end{table*}")

for l in lines:
    print(l)
