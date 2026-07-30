"""Generate LaTeX table for CEM/GC-CEM CTL and ICL (KSG probability space)."""
import joblib
import numpy as np
import os

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"


def fmt(mean, std):
    return f"${mean:.2f}_{{{chr(92)}pm {std:.2f}}}$"


d_tt = joblib.load(master + "results/results_tabulartoy_acem_suite.dict")
d_ds = joblib.load(master + "results/results_dsprites_suite.dict")

lams = ["lam_c0.1", "lam_c0.5", "lam_c1"]
lam_labels = ["0.1", "0.5", "1.0"]

DATASETS = [
    ("TabularToy", d_tt, [("cem", "CEM"), ("acem", "GC-CEM")]),
    ("dSprites",   d_ds, [("cem", "CEM"), ("crcem", "GC-CEM")]),
]

rows = []
for dataset_label, d, model_map in DATASETS:
    first_dataset = True
    for model_key, model_label in model_map:
        first_model = True
        for lam, lam_label in zip(lams, lam_labels):
            ctls, icls, tasks = [], [], []
            for fk, r in d.get(model_key, {}).get(lam, {}).items():
                cm = r.get("ctl_mean")
                if cm is not None:
                    ctls.append(float(np.mean(cm)))
                v = r.get("icl_mean", float("nan"))
                if not np.isnan(v):
                    icls.append(v)
                tasks.append(r.get("task_acc", float("nan")))
            if not ctls:
                continue

            ds_col  = f"\\multirow{{6}}{{*}}{{{dataset_label}}}" if first_dataset else ""
            mod_col = f"\\multirow{{3}}{{*}}{{{model_label}}}" if first_model else ""
            task_s  = fmt(np.nanmean(tasks) * 100, np.nanstd(tasks) * 100)
            ctl_s   = fmt(np.nanmean(ctls), np.nanstd(ctls))
            icl_s   = fmt(np.nanmean(icls), np.nanstd(icls))
            rows.append(f"  {ds_col} & {mod_col} & {lam_label} & {task_s} & {ctl_s} & {icl_s} \\\\")
            first_dataset = False
            first_model = False
    rows.append("\\midrule")

rows[-1] = "\\bottomrule"

table = "\n".join([
    r"\begin{table}[h]",
    r"\centering",
    r"\caption{CTL and ICL (KSG, probability space) for CEM and GC-CEM.}",
    r"\label{tab:cem_ctl_icl}",
    r"\begin{tabular}{llcrrr}",
    r"\toprule",
    r"Dataset & Model & $\lambda_c$ & Task Acc (\%) & CTL$\downarrow$ & ICL$\downarrow$ \\",
    r"\midrule",
] + rows + [
    r"\end{tabular}",
    r"\end{table}",
])

print(table)
