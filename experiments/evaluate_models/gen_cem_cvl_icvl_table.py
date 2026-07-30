"""Generate LaTeX table for CEM/GC-CEM CVL and ICVL across datasets."""
import joblib
import numpy as np
import os

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"


def get_stats(d, model_key, lam, metric_fn):
    vals = []
    for fk, r in d.get(model_key, {}).get(lam, {}).items():
        v = metric_fn(r)
        if v is not None and not np.isnan(float(v)):
            vals.append(float(v))
    return (np.mean(vals), np.std(vals), len(vals)) if vals else (float("nan"), float("nan"), 0)


def fmt(mean, std):
    return f"${mean:.2f}_{{{chr(92)}pm {std:.2f}}}$"


d_tt  = joblib.load(master + "results/results_tabulartoy_acem_suite.dict")
d_ds  = joblib.load(master + "results/results_dsprites_suite.dict")
d_cub = joblib.load(master + "results/results_cub_suite.dict")

lams       = ["lam_c0.1", "lam_c0.5", "lam_c1"]
lam_labels = ["0.1", "0.5", "1.0"]

DATASETS = [
    ("TabularToy", d_tt,  [("cem", "CEM"), ("acem",  "GC-CEM")], lams,       lam_labels),
    ("dSprites",   d_ds,  [("cem", "CEM"), ("crcem", "GC-CEM")], lams,       lam_labels),
    ("CUB",        d_cub, [("cem", "CEM"), ("crcem", "GC-CEM")], ["lam_c0.1"], ["0.1"]),
]

rows = []
for dataset_label, d, model_map, ds_lams, ds_lam_labels in DATASETS:
    n_rows = len(model_map) * len(ds_lams)
    first_dataset = True
    for model_key, model_label in model_map:
        first_model = True
        for lam, lam_label in zip(ds_lams, ds_lam_labels):
            cvl_m,  cvl_s,  n = get_stats(d, model_key, lam, lambda r: r.get("cvl",  {}).get("CVL"))
            icvl_m, icvl_s, _ = get_stats(d, model_key, lam, lambda r: r.get("icvl", {}).get("ICVL"))
            task_m, task_s, _ = get_stats(d, model_key, lam, lambda r: r.get("task_acc"))
            if n == 0:
                continue

            ds_col  = f"\\multirow{{{n_rows}}}{{*}}{{{dataset_label}}}" if first_dataset else ""
            mod_col = f"\\multirow{{{len(ds_lams)}}}{{*}}{{{model_label}}}" if first_model else ""
            task_s_fmt = fmt(task_m * 100, task_s * 100)
            cvl_fmt  = fmt(cvl_m,  cvl_s)
            icvl_fmt = fmt(icvl_m, icvl_s)
            rows.append(f"  {ds_col} & {mod_col} & {lam_label} & {task_s_fmt} & {cvl_fmt} & {icvl_fmt} \\\\")
            first_dataset = False
            first_model   = False
    rows.append("\\midrule")

rows[-1] = "\\bottomrule"

table = "\n".join([
    r"\begin{table}[h]",
    r"\centering",
    r"\caption{CVL and ICVL for CEM and GC-CEM across datasets. CUB reports $\lambda_c=0.1$ only.}",
    r"\label{tab:cem_cvl_icvl}",
    r"\begin{tabular}{llcrrr}",
    r"\toprule",
    r"Dataset & Model & $\lambda_c$ & Task Acc (\%) & CVL$\downarrow$ & ICVL$\downarrow$ \\",
    r"\midrule",
] + rows + [
    r"\end{tabular}",
    r"\end{table}",
])

print(table)
