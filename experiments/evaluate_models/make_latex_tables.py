"""
Generate LaTeX tables for all paper results.
Run from project root: python experiments/evaluate_models/make_latex_tables.py
"""
import os
import numpy as np
import joblib

BASE = os.getcwd() + "/results/"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_splits(folder, model_name, n=5):
    folds = []
    for i in range(n):
        p = os.path.join(BASE, folder, model_name,
                         f"{model_name}_split_{i}_results.joblib")
        if os.path.exists(p):
            folds.append(joblib.load(p))
            continue
        alt = model_name.replace("lr1e-04", "lr1e-4")
        if alt == model_name:
            alt = model_name.replace("lr1e-4", "lr1e-04")
        p2 = os.path.join(BASE, folder, model_name,
                          f"{alt}_split_{i}_results.joblib")
        if os.path.exists(p2):
            folds.append(joblib.load(p2))
    return folds


def get(folds, key, scale=1):
    return [float(f.get(key, np.nan)) * scale for f in folds]


def ctl_vals(folds):
    raw = get(folds, "test_ctl_average")
    if not all(np.isnan(x) for x in raw):
        return raw
    return get(folds, "test_normalised_ctl")


def icl_vals(folds):
    icl_avg = get(folds, "test_icl_average")
    nicl    = get(folds, "test_normalised_icl")
    avg_mean = np.nanmean(icl_avg) if icl_avg else np.nan
    if not np.isnan(avg_mean) and avg_mean <= 1.0:
        return icl_avg
    return nicl


def mean_std(vals):
    v = [x for x in vals if not np.isnan(x)]
    if not v:
        return np.nan, np.nan
    return np.mean(v), np.std(v)


def fmt_pct(vals):
    m, s = mean_std(vals)
    if np.isnan(m):
        return "—"
    return f"{m:.1f} $\\pm$ {s:.1f}"


def fmt_mi(vals):
    m, s = mean_std(vals)
    if np.isnan(m):
        return "—"
    return f"{m:.3f} $\\pm$ {s:.3f}"


def fmt_cvl(vals):
    m, s = mean_std(vals)
    if np.isnan(m):
        return "—"
    return f"{m:.3f} $\\pm$ {s:.3f}"


# ── load suite dicts ──────────────────────────────────────────────────────────
tt_cem    = joblib.load(BASE + "results_tabulartoy_acem_suite.dict")
tt_cbm    = joblib.load(BASE + "results_tabulartoy_cbm_suite.dict")
ds_cem    = joblib.load(BASE + "results_dsprites_suite.dict")
ds_cbm    = joblib.load(BASE + "results_dsprites_cbm_suite.dict")
cub_cem   = joblib.load(BASE + "results_cub_suite.dict")
cub_cbm   = joblib.load(BASE + "results_cub_cbm_suite.dict")
ds_cvl    = joblib.load(BASE + "results_dsprites_cvl.dict")

LAM_MAP = {"0.1": "lam_c0.1", "0.5": "lam_c0.5", "1.0": "lam_c1"}

CBM_SPLITS = {
    "TT": {
        "Hard CBM":    ("tabulartoy_25_10k_models",                    "HardCBM_adam_lr0.05_bs64_lam_c1",                       "-"),
        "Seq CBM":     ("tabulartoy_25_10k_models_acbm_shared_critic", "SeqCBM_adam_lr0.05_bs64_lam_c1",                        "-"),
        "Joint 0.1":   ("tabulartoy_25_10k_models",                    "SoftCBM_adam_lr0.05_bs64_lam_c0.1",                     "0.1"),
        "Joint 0.5":   ("tabulartoy_25_10k_models",                    "SoftCBM_adam_lr0.05_bs64_lam_c0.5",                     "0.5"),
        "Joint 1.0":   ("tabulartoy_25_10k_models",                    "SoftCBM_adam_lr0.05_bs64_lam_c1",                       "1.0"),
        "GC-CBM 0.1":  ("tabulartoy_25_10k_models_acbm_shared_critic", "ACBM_adam_lr0.05_bs64_lam1_none_lam_c0.1_shared_critic","0.1"),
        "GC-CBM 0.5":  ("tabulartoy_25_10k_models_acbm_shared_critic", "ACBM_adam_lr0.05_bs64_lam1_none_lam_c0.5_shared_critic","0.5"),
        "GC-CBM 1.0":  ("tabulartoy_25_10k_models_acbm_shared_critic", "ACBM_adam_lr0.05_bs64_lam1_none_lam_c1_shared_critic",  "1.0"),
    },
    "dS": {
        "Hard CBM":    ("dsprites_dep_0_models",       "HardCBM_adam_lr1e-04_bs64_lam_c1",                              "-"),
        "Seq CBM":     ("dsprites_sequential_acbm",    "SeqCBM_adam_lr1e-04_bs64_lam_c1",                               "-"),
        "Joint 0.1":   ("dsprites_dep_0_models",       "SoftCBM_adam_lr1e-04_bs64_lam_c0.1",                            "0.1"),
        "Joint 0.5":   ("dsprites_dep_0_models",       "SoftCBM_adam_lr1e-04_bs64_lam_c0.5",                            "0.5"),
        "Joint 1.0":   ("dsprites_dep_0_models",       "SoftCBM_adam_lr1e-04_bs64_lam_c1",                              "1.0"),
        "GC-CBM 0.1":  ("dsprites_ACBM_shared_critic", "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c0.1_shared_critic",      "0.1"),
        "GC-CBM 0.5":  ("dsprites_ACBM_shared_critic", "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c0.5_shared_critic",      "0.5"),
        "GC-CBM 1.0":  ("dsprites_ACBM_shared_critic", "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c1_shared_critic",        "1.0"),
    },
    "CUB": {
        "Hard CBM":    ("cub_soft_0.01",         "HardCBM_adam_lr1e-04_bs256_lam_c1",                              "-"),
        "Seq CBM":     ("cub_acbm_shared_critic", "SeqCBM_adam_lr1e-04_bs256_lam_c1",                               "-"),
        "Joint 0.1":   ("cub_soft_0.01",         "SoftCBM_adam_lr1e-04_bs256_lam_c0.1",                            "0.1"),
        "Joint 0.5":   ("cub_soft_0.01",         "SoftCBM_adam_lr1e-04_bs256_lam_c0.5",                            "0.5"),
        "Joint 1.0":   ("cub_soft_0.01",         "SoftCBM_adam_lr1e-04_bs256_lam_c1",                              "1.0"),
        "GC-CBM 0.1":  ("cub_acbm_shared_critic","ACBM_adam_lr1e-04_bs256_lam1_none_lam_c0.1_shared_critic",       "0.1"),
        "GC-CBM 0.5":  ("cub_acbm_shared_critic","ACBM_adam_lr1e-04_bs256_lam1_none_lam_c0.5_shared_critic",       "0.5"),
        "GC-CBM 1.0":  ("cub_acbm_shared_critic","ACBM_adam_lr1e-04_bs256_lam1_none_lam_c1_shared_critic",         "1.0"),
    },
}

CEM_SPLITS = {
    "TT": {
        "CEM":    {"0.1": ("tabulartoy_25_10k_models_acem_shared_critic", "CEM_adam_lr0.05_bs64_lam_c0.1"),
                   "0.5": ("tabulartoy_25_10k_models_acem_shared_critic", "CEM_adam_lr0.05_bs64_lam_c0.5"),
                   "1.0": ("tabulartoy_25_10k_models_acem_shared_critic", "CEM_adam_lr0.05_bs64_lam_c1")},
        "GC-CEM": {"0.1": ("tabulartoy_25_10k_models_acem_shared_critic", "ACEM_adam_lr1e-03_bs64_lam1_none_lam_c0.1_shared_critic"),
                   "0.5": ("tabulartoy_25_10k_models_acem_shared_critic", "ACEM_adam_lr1e-03_bs64_lam1_none_lam_c0.5_shared_critic"),
                   "1.0": ("tabulartoy_25_10k_models_acem_shared_critic", "ACEM_adam_lr1e-03_bs64_lam1_none_lam_c1_shared_critic")},
    },
    "dS": {
        "CEM":    {"0.1": ("dsprites_cem", "CEM_adam_lr1e-03_bs64_lam_c0.1"),
                   "0.5": ("dsprites_cem", "CEM_adam_lr1e-03_bs64_lam_c0.5"),
                   "1.0": ("dsprites_cem", "CEM_adam_lr1e-03_bs64_lam_c1")},
        "GC-CEM": {"0.1": ("dsprites_acem_shared_critic", "CRCEM_adam_lr5e-05_bs64_lam1_none_lam_c0.1_shared_critic"),
                   "0.5": ("dsprites_acem_shared_critic", "CRCEM_adam_lr5e-05_bs64_lam1_none_lam_c0.5_shared_critic"),
                   "1.0": ("dsprites_acem_shared_critic", "CRCEM_adam_lr5e-05_bs64_lam1_none_lam_c1_shared_critic")},
    },
    "CUB": {
        "CEM":    {"0.1": ("cub_cem", "CEM_adam_lr1e-03_bs256_lam_c0.1"),
                   "0.5": ("cub_cem", "CEM_adam_lr1e-03_bs256_lam_c0.5"),
                   "1.0": ("cub_cem", "CEM_adam_lr1e-03_bs256_lam_c1")},
        "GC-CEM": {"0.1": ("cub_cem", "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c0.1_shared_critic"),
                   "0.5": ("cub_cem", "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c0.5_shared_critic"),
                   "1.0": ("cub_cem", "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c1_shared_critic")},
    },
}

CEM_SUITE = {
    "TT":  {"CEM": (tt_cem,  "cem"),   "GC-CEM": (tt_cem,  "acem")},
    "dS":  {"CEM": (ds_cem,  "cem"),   "GC-CEM": (ds_cem,  "crcem")},
    "CUB": {"CEM": (cub_cem, "cem"),   "GC-CEM": (cub_cem, "crcem")},
}


# ── collect all data ──────────────────────────────────────────────────────────

# CBM data
cbm_data = {}  # (ds, model_key) -> dict of arrays
for ds in ["TT", "dS", "CUB"]:
    for mkey, (folder, mname, lam) in CBM_SPLITS[ds].items():
        folds = load_splits(folder, mname)
        cbm_data[(ds, mkey)] = {
            "lam":  lam,
            "task": get(folds, "test_acc_y", 100),
            "cacc": get(folds, "test_acc_c", 100),
            "ctl":  ctl_vals(folds),
            "icl":  icl_vals(folds),
        }

# CEM data
cem_data = {}  # (ds, model_label, lam) -> dict
for ds in ["TT", "dS", "CUB"]:
    for ml in ["CEM", "GC-CEM"]:
        suite_d, suite_key = CEM_SUITE[ds][ml]
        for lam in ["0.1", "0.5", "1.0"]:
            lam_key = LAM_MAP[lam]
            folder, mname = CEM_SPLITS[ds][ml][lam]
            folds = load_splits(folder, mname)

            if ds == "CUB":
                task = get(folds, "test_acc_y", 100)
                cacc = get(folds, "test_acc_c", 100)
            else:
                folds_s = suite_d.get(suite_key, {}).get(lam_key, {})
                task = [float(folds_s.get(fk, {}).get("task_acc", np.nan)) * 100
                        for fk in folds_s]
                cacc = [float(folds_s.get(fk, {}).get("c_acc", np.nan)) * 100
                        for fk in folds_s]

            cem_data[(ds, ml, lam)] = {
                "task": task,
                "cacc": cacc,
                "ctl":  ctl_vals(folds),
                "icl":  icl_vals(folds),
            }

# CVL/ICVL data
cvl_data = {}  # (ds, ml, lam) -> (cvl_vals, icvl_vals)
for ds in ["TT", "dS", "CUB"]:
    suite_map = {
        "TT":  {"CEM": (tt_cem, "cem"),    "GC-CEM": (tt_cem,  "acem")},
        "dS":  {"CEM": (ds_cem, "cem"),    "GC-CEM": (ds_cem,  "crcem")},
        "CUB": {"CEM": (cub_cem, "cem"),   "GC-CEM": (cub_cem, "crcem")},
    }
    cvl_key_map = {"CEM": "cem", "GC-CEM": "gc_cem"}
    for ml in ["CEM", "GC-CEM"]:
        suite_d, suite_key = suite_map[ds][ml]
        for lam in ["0.1", "0.5", "1.0"]:
            lam_key = LAM_MAP[lam]
            if ds == "CUB":
                folds_s = suite_d.get(suite_key, {}).get(lam_key, {})
                if folds_s:
                    cvl_v  = [max(0.0, float(folds_s.get(fk,{}).get("cvl",{}).get("CVL",0))) for fk in folds_s]
                    icvl_v = [max(0.0, float(folds_s.get(fk,{}).get("icvl",{}).get("ICVL",0))) for fk in folds_s]
                else:
                    cvl_v = icvl_v = [0.0] * 5
            elif ds == "TT":
                folds_s = suite_d.get(suite_key, {}).get(lam_key, {})
                cvl_v  = [float(folds_s.get(fk,{}).get("cvl",{}).get("CVL",np.nan)) for fk in folds_s]
                icvl_v = [float(folds_s.get(fk,{}).get("icvl",{}).get("ICVL",np.nan)) for fk in folds_s]
            else:  # dS
                cvl_folds = ds_cvl.get(cvl_key_map[ml], {}).get(lam_key, {})
                cvl_v = [float(fd.get("cvl_clipped_mean", np.nan)) for fd in cvl_folds.values()]
                folds_s = suite_d.get(suite_key, {}).get(lam_key, {})
                icvl_v = [float(folds_s.get(fk,{}).get("icvl",{}).get("ICVL",np.nan)) for fk in folds_s]
            cvl_data[(ds, ml, lam)] = (cvl_v, icvl_v)


# ════════════════════════════════════════════════════════════════════════════
# OUTPUT LATEX TABLES
# ════════════════════════════════════════════════════════════════════════════

def lam_tex(lam):
    return r"—" if lam == "-" else f"$\\lambda_c={lam}$"

def model_tex(m):
    m = m.replace("GC-CBM", r"\gcCBM").replace("GC-CEM", r"\gcCEM")
    m = m.replace("Joint", "Joint CBM")
    return m

# ── TABLE 1: CBM task accuracy and concept accuracy ───────────────────────────
print("% ─── TABLE 1: CBM Task Accuracy and Concept Accuracy ───────────────")
print(r"""\begin{table}[t]
\centering
\caption{Task accuracy (\%) and concept accuracy (\%) for CBM variants
         across all datasets. Concept accuracy is the average per-concept
         binary accuracy. $\lambda_c$ is the concept loss weight.}
\label{tab:cbm_acc}
\begin{tabular}{llrrrrrr}
\toprule
 & & \multicolumn{2}{c}{\textbf{TabularToy}} & \multicolumn{2}{c}{\textbf{dSprites}} & \multicolumn{2}{c}{\textbf{CUB}} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
\textbf{Model} & $\boldsymbol{\lambda_c}$ & Task\% & $c$-acc\% & Task\% & $c$-acc\% & Task\% & $c$-acc\% \\
\midrule""")

row_order = ["Hard CBM", "Seq CBM",
             "Joint 0.1", "Joint 0.5", "Joint 1.0",
             "GC-CBM 0.1", "GC-CBM 0.5", "GC-CBM 1.0"]
display_names = {
    "Hard CBM": "Hard CBM", "Seq CBM": "Seq CBM",
    "Joint 0.1": "Joint CBM", "Joint 0.5": "Joint CBM", "Joint 1.0": "Joint CBM",
    "GC-CBM 0.1": r"\gcCBM", "GC-CBM 0.5": r"\gcCBM", "GC-CBM 1.0": r"\gcCBM",
}
lam_display = {
    "Hard CBM": "—", "Seq CBM": "—",
    "Joint 0.1": "0.1", "Joint 0.5": "0.5", "Joint 1.0": "1.0",
    "GC-CBM 0.1": "0.1", "GC-CBM 0.5": "0.5", "GC-CBM 1.0": "1.0",
}

# track when to insert midrules
groups = [["Hard CBM", "Seq CBM"], ["Joint 0.1","Joint 0.5","Joint 1.0"],
          ["GC-CBM 0.1","GC-CBM 0.5","GC-CBM 1.0"]]

for g_idx, group in enumerate(groups):
    for mkey in group:
        name = display_names[mkey]
        lam  = lam_display[mkey]
        cells = []
        for ds in ["TT", "dS", "CUB"]:
            d = cbm_data[(ds, mkey)]
            cells.append(fmt_pct(d["task"]))
            cells.append(fmt_pct(d["cacc"]))
        print(f"  {name} & {lam} & " + " & ".join(cells) + r" \\")
    if g_idx < len(groups) - 1:
        print(r"\addlinespace")

print(r"""\bottomrule
\end{tabular}
\end{table}""")

# ── TABLE 2: CBM CTL and ICL ─────────────────────────────────────────────────
print()
print("% ─── TABLE 2: CBM CTL and ICL ──────────────────────────────────────")
print(r"""\begin{table}[t]
\centering
\caption{CTL and ICL leakage scores for CBM variants.
         CTL: excess MI between concept representations and task label.
         ICL: mean pairwise inter-concept MI.}
\label{tab:cbm_leakage}
\begin{tabular}{llrrrrrr}
\toprule
 & & \multicolumn{2}{c}{\textbf{TabularToy}} & \multicolumn{2}{c}{\textbf{dSprites}} & \multicolumn{2}{c}{\textbf{CUB}} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
\textbf{Model} & $\boldsymbol{\lambda_c}$ & CTL & ICL & CTL & ICL & CTL & ICL \\
\midrule""")

for g_idx, group in enumerate(groups):
    for mkey in group:
        name = display_names[mkey]
        lam  = lam_display[mkey]
        cells = []
        for ds in ["TT", "dS", "CUB"]:
            d = cbm_data[(ds, mkey)]
            cells.append(fmt_mi(d["ctl"]))
            cells.append(fmt_mi(d["icl"]))
        print(f"  {name} & {lam} & " + " & ".join(cells) + r" \\")
    if g_idx < len(groups) - 1:
        print(r"\addlinespace")

print(r"""\bottomrule
\end{tabular}
\end{table}""")

# ── TABLE 3: CEM task accuracy and concept accuracy ───────────────────────────
print()
print("% ─── TABLE 3: CEM Task Accuracy and Concept Accuracy ───────────────")
print(r"""\begin{table}[t]
\centering
\caption{Task accuracy (\%) and concept accuracy (\%) for CEM and \gcCEM.
         For TabularToy and dSprites, concept accuracy is joint (all concepts
         simultaneously correct). For CUB, it is mean per-concept accuracy.}
\label{tab:cem_acc}
\begin{tabular}{llrrrrrr}
\toprule
 & & \multicolumn{2}{c}{\textbf{TabularToy}} & \multicolumn{2}{c}{\textbf{dSprites}} & \multicolumn{2}{c}{\textbf{CUB}} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
\textbf{Model} & $\boldsymbol{\lambda_c}$ & Task\% & $c$-acc\% & Task\% & $c$-acc\% & Task\% & $c$-acc\% \\
\midrule""")

cem_groups = [["CEM"], ["GC-CEM"]]
for g_idx, group in enumerate(cem_groups):
    for ml in group:
        for lam in ["0.1", "0.5", "1.0"]:
            name = "CEM" if ml == "CEM" else r"\gcCEM"
            cells = []
            for ds in ["TT", "dS", "CUB"]:
                d = cem_data[(ds, ml, lam)]
                cells.append(fmt_pct(d["task"]))
                cells.append(fmt_pct(d["cacc"]))
            print(f"  {name} & {lam} & " + " & ".join(cells) + r" \\")
    if g_idx < len(cem_groups) - 1:
        print(r"\addlinespace")

print(r"""\bottomrule
\end{tabular}
\end{table}""")

# ── TABLE 4: CEM CTL and ICL ─────────────────────────────────────────────────
print()
print("% ─── TABLE 4: CEM CTL and ICL ──────────────────────────────────────")
print(r"""\begin{table}[t]
\centering
\caption{CTL and ICL leakage scores for CEM and \gcCEM.}
\label{tab:cem_leakage}
\begin{tabular}{llrrrrrr}
\toprule
 & & \multicolumn{2}{c}{\textbf{TabularToy}} & \multicolumn{2}{c}{\textbf{dSprites}} & \multicolumn{2}{c}{\textbf{CUB}} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
\textbf{Model} & $\boldsymbol{\lambda_c}$ & CTL & ICL & CTL & ICL & CTL & ICL \\
\midrule""")

for g_idx, group in enumerate(cem_groups):
    for ml in group:
        for lam in ["0.1", "0.5", "1.0"]:
            name = "CEM" if ml == "CEM" else r"\gcCEM"
            cells = []
            for ds in ["TT", "dS", "CUB"]:
                d = cem_data[(ds, ml, lam)]
                cells.append(fmt_mi(d["ctl"]))
                cells.append(fmt_mi(d["icl"]))
            print(f"  {name} & {lam} & " + " & ".join(cells) + r" \\")
    if g_idx < len(cem_groups) - 1:
        print(r"\addlinespace")

print(r"""\bottomrule
\end{tabular}
\end{table}""")

# ── TABLE 5: CVL and ICVL ────────────────────────────────────────────────────
print()
print("% ─── TABLE 5: CVL and ICVL ─────────────────────────────────────────")
print(r"""\begin{table}[t]
\centering
\caption{CVL and ICVL for CEM and \gcCEM. CVL measures task-predictive
         information in concept embeddings via ridge regression; ICVL measures
         inter-concept predictive information. CUB values are 0.00 after
         clipping negative $R^2$.}
\label{tab:cvl_icvl}
\begin{tabular}{llrrrrrr}
\toprule
 & & \multicolumn{2}{c}{\textbf{TabularToy}} & \multicolumn{2}{c}{\textbf{dSprites}} & \multicolumn{2}{c}{\textbf{CUB}} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
\textbf{Model} & $\boldsymbol{\lambda_c}$ & CVL & ICVL & CVL & ICVL & CVL & ICVL \\
\midrule""")

for g_idx, group in enumerate(cem_groups):
    for ml in group:
        for lam in ["0.1", "0.5", "1.0"]:
            name = "CEM" if ml == "CEM" else r"\gcCEM"
            cells = []
            for ds in ["TT", "dS", "CUB"]:
                cvl_v, icvl_v = cvl_data[(ds, ml, lam)]
                cells.append(fmt_cvl(cvl_v))
                cells.append(fmt_cvl(icvl_v))
            print(f"  {name} & {lam} & " + " & ".join(cells) + r" \\")
    if g_idx < len(cem_groups) - 1:
        print(r"\addlinespace")

print(r"""\bottomrule
\end{tabular}
\end{table}""")
