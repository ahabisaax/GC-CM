"""
Rigorous verification of all saved results against paper tables.
Run from project root: python experiments/evaluate_models/paper_verify.py
"""
import os
import numpy as np
import joblib

BASE = os.getcwd() + "/results/"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_splits(folder, model_name, n=5):
    """Load split results; tries alternate lr1e-4 vs lr1e-04 naming."""
    folds = []
    for i in range(n):
        p = os.path.join(BASE, folder, model_name,
                         f"{model_name}_split_{i}_results.joblib")
        if os.path.exists(p):
            folds.append((i, joblib.load(p)))
            continue
        # try alternate naming (lr1e-04 ↔ lr1e-4)
        alt = model_name.replace("lr1e-04", "lr1e-4")
        if alt == model_name:
            alt = model_name.replace("lr1e-4", "lr1e-04")
        p2 = os.path.join(BASE, folder, model_name,
                          f"{alt}_split_{i}_results.joblib")
        if os.path.exists(p2):
            folds.append((i, joblib.load(p2)))
    return [d for _, d in sorted(folds)]


def s(vals, scale=1):
    v = [float(x) * scale for x in vals if not np.isnan(float(x))]
    if not v:
        return "   —  ±  —  "
    return f"{np.mean(v):.3f}±{np.std(v):.3f}"


def get(folds, key, scale=1):
    return [float(f.get(key, np.nan)) * scale for f in folds]


def ctl_vals(folds):
    """CTL: prefer test_ctl_average; fallback to test_normalised_ctl."""
    raw = get(folds, "test_ctl_average")
    if not all(np.isnan(x) for x in raw):
        return raw
    return get(folds, "test_normalised_ctl")


def icl_vals(folds):
    """ICL: use test_icl_average when ≤1.0; else test_normalised_icl."""
    icl_avg  = get(folds, "test_icl_average")
    nicl     = get(folds, "test_normalised_icl")
    avg_mean = np.nanmean(icl_avg) if icl_avg else np.nan
    if not np.isnan(avg_mean) and avg_mean <= 1.0:
        return icl_avg
    return nicl


# ── suite dicts ──────────────────────────────────────────────────────────────
tt_cem  = joblib.load(BASE + "results_tabulartoy_acem_suite.dict")
tt_cbm  = joblib.load(BASE + "results_tabulartoy_cbm_suite.dict")
ds_cem  = joblib.load(BASE + "results_dsprites_suite.dict")
ds_cbm  = joblib.load(BASE + "results_dsprites_cbm_suite.dict")
cub_cem = joblib.load(BASE + "results_cub_suite.dict")
cub_cbm = joblib.load(BASE + "results_cub_cbm_suite.dict")
tt_onehot = joblib.load(BASE + "results_tabulartoy_cvl_onehot.dict")
ds_onehot = joblib.load(BASE + "results_dsprites_cvl_onehot.dict")

SEP  = "=" * 120
HSEP = "-" * 120

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 & 2: CBM performance + CTL/ICL
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("TAB 1 & 2: CBM TASK/CONCEPT ACC + CTL/ICL")
print(f"{'Model':<15} {'λ':<5} {'DS':<4} | {'task%':>12} {'c_acc%':>12} {'CTL':>12} {'ICL':>12} | {'p_task':>7} {'p_c':>6} {'p_CTL':>7} {'p_ICL':>7}")
print(HSEP)

CBM_SPLITS = {
    "TT": {
        "Hard CBM":      ("tabulartoy_25_10k_models",                    "HardCBM_adam_lr0.05_bs64_lam_c1",                        "-"),
        "Seq CBM":       ("tabulartoy_25_10k_models_acbm_shared_critic", "SeqCBM_adam_lr0.05_bs64_lam_c1",                         "-"),
        "Joint CBM 0.1": ("tabulartoy_25_10k_models",                    "SoftCBM_adam_lr0.05_bs64_lam_c0.1",                      "0.1"),
        "Joint CBM 0.5": ("tabulartoy_25_10k_models",                    "SoftCBM_adam_lr0.05_bs64_lam_c0.5",                      "0.5"),
        "Joint CBM 1.0": ("tabulartoy_25_10k_models",                    "SoftCBM_adam_lr0.05_bs64_lam_c1",                        "1.0"),
        "GC-CBM 0.1":    ("tabulartoy_25_10k_models_acbm_shared_critic", "ACBM_adam_lr0.05_bs64_lam1_none_lam_c0.1_shared_critic",  "0.1"),
        "GC-CBM 0.5":    ("tabulartoy_25_10k_models_acbm_shared_critic", "ACBM_adam_lr0.05_bs64_lam1_none_lam_c0.5_shared_critic",  "0.5"),
        "GC-CBM 1.0":    ("tabulartoy_25_10k_models_acbm_shared_critic", "ACBM_adam_lr0.05_bs64_lam1_none_lam_c1_shared_critic",    "1.0"),
    },
    "dS": {
        "Hard CBM":      ("dsprites_dep_0_models",       "HardCBM_adam_lr1e-04_bs64_lam_c1",                              "-"),
        "Seq CBM":       ("dsprites_sequential_acbm",    "SeqCBM_adam_lr1e-04_bs64_lam_c1",                               "-"),
        "Joint CBM 0.1": ("dsprites_dep_0_models",       "SoftCBM_adam_lr1e-04_bs64_lam_c0.1",                            "0.1"),
        "Joint CBM 0.5": ("dsprites_dep_0_models",       "SoftCBM_adam_lr1e-04_bs64_lam_c0.5",                            "0.5"),
        "Joint CBM 1.0": ("dsprites_dep_0_models",       "SoftCBM_adam_lr1e-04_bs64_lam_c1",                              "1.0"),
        "GC-CBM 0.1":    ("dsprites_ACBM_shared_critic", "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c0.1_shared_critic",      "0.1"),
        "GC-CBM 0.5":    ("dsprites_ACBM_shared_critic", "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c0.5_shared_critic",      "0.5"),
        "GC-CBM 1.0":    ("dsprites_ACBM_shared_critic", "CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c1_shared_critic",        "1.0"),
    },
    "CUB": {
        "Hard CBM":      ("cub_soft_0.01",         "HardCBM_adam_lr1e-04_bs256_lam_c1",                               "-"),
        "Seq CBM":       ("cub_acbm_shared_critic", "SeqCBM_adam_lr1e-04_bs256_lam_c1",                                "-"),
        "Joint CBM 0.1": ("cub_soft_0.01",         "SoftCBM_adam_lr1e-04_bs256_lam_c0.1",                             "0.1"),
        "Joint CBM 0.5": ("cub_soft_0.01",         "SoftCBM_adam_lr1e-04_bs256_lam_c0.5",                             "0.5"),
        "Joint CBM 1.0": ("cub_soft_0.01",         "SoftCBM_adam_lr1e-04_bs256_lam_c1",                               "1.0"),
        "GC-CBM 0.1":    ("cub_acbm_shared_critic", "ACBM_adam_lr1e-04_bs256_lam1_none_lam_c0.1_shared_critic",       "0.1"),
        "GC-CBM 0.5":    ("cub_acbm_shared_critic", "ACBM_adam_lr1e-04_bs256_lam1_none_lam_c0.5_shared_critic",       "0.5"),
        "GC-CBM 1.0":    ("cub_acbm_shared_critic", "ACBM_adam_lr1e-04_bs256_lam1_none_lam_c1_shared_critic",         "1.0"),
    },
}

# paper reference  (task%, c_acc%, CTL, ICL)
PAPER_CBM = {
    "TT": {
        "Hard CBM":      (99.0, 99.3, 0.032, 0.168),
        "Seq CBM":       (99.4, 99.4, 0.034, 0.028),
        "Joint CBM 0.1": (99.4, 55.5, 0.492, 0.379),
        "Joint CBM 0.5": (99.5, 87.6, 0.418, 0.300),
        "Joint CBM 1.0": (99.4, 95.8, 0.355, 0.198),
        "GC-CBM 0.1":    (99.4, 99.4, 0.073, 0.022),
        "GC-CBM 0.5":    (99.2, 99.3, 0.062, 0.013),
        "GC-CBM 1.0":    (99.4, 99.6, 0.113, 0.030),
    },
    "dS": {
        "Hard CBM":      (95.8, 99.1, 0.044, 0.048),
        "Seq CBM":       (97.4, 99.5, 0.005, 0.012),
        "Joint CBM 0.1": (97.7, 97.3, 0.277, 0.408),
        "Joint CBM 0.5": (97.4, 99.4, 0.079, 0.223),
        "Joint CBM 1.0": (97.2, 99.4, 0.063, 0.181),
        "GC-CBM 0.1":    (96.2, 99.2, 0.008, 0.008),
        "GC-CBM 0.5":    (96.6, 99.3, 0.007, 0.009),
        "GC-CBM 1.0":    (97.0, 99.4, 0.006, 0.008),
    },
    "CUB": {
        "Hard CBM":      (61.8, 94.4, 0.034, 0.005),
        "Seq CBM":       (65.9, 94.7, 0.029, 0.008),
        "Joint CBM 0.1": (71.0, 71.1, 0.043, 0.009),
        "Joint CBM 0.5": (72.4, 86.5, 0.044, 0.012),
        "Joint CBM 1.0": (68.0, 91.2, 0.032, 0.013),
        "GC-CBM 0.1":    (65.9, 94.7, 0.024, 0.010),
        "GC-CBM 0.5":    (66.1, 94.6, 0.023, 0.009),
        "GC-CBM 1.0":    (66.5, 94.8, 0.026, 0.007),
    },
}

for ds_label in ["TT", "dS", "CUB"]:
    for model_key, (folder, model_name, lam) in CBM_SPLITS[ds_label].items():
        folds = load_splits(folder, model_name)
        if not folds:
            task_s = c_s = ctl_s = icl_s = "MISSING"
        else:
            task_s = s(get(folds, "test_acc_y", 100))
            c_s    = s(get(folds, "test_acc_c", 100))
            ctl_s  = s(ctl_vals(folds))
            icl_s  = s(icl_vals(folds))

        pt, pc, pctl, picl = PAPER_CBM[ds_label].get(model_key, (np.nan,) * 4)
        print(f"{model_key:<15} {lam:<5} {ds_label:<4} | {task_s:>12} {c_s:>12} {ctl_s:>12} {icl_s:>12} | {pt:>7.1f} {pc:>6.1f} {pctl:>7.3f} {picl:>7.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3 & 4: CEM/GC-CEM accuracy + CTL/ICL
# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("TAB 3 & 4: CEM/GC-CEM TASK/CONCEPT ACC + CTL/ICL")
print(f"{'Model':<12} {'λ':<5} {'DS':<4} | {'task%':>12} {'c_acc%':>12} {'CTL':>12} {'ICL':>12} | {'p_task':>7} {'p_c':>6} {'p_CTL':>7} {'p_ICL':>7}")
print(HSEP)

CEM_SPLITS = {
    "TT": {
        "CEM": {
            "0.1": ("tabulartoy_25_10k_models_acem_shared_critic", "CEM_adam_lr0.05_bs64_lam_c0.1"),
            "0.5": ("tabulartoy_25_10k_models_acem_shared_critic", "CEM_adam_lr0.05_bs64_lam_c0.5"),
            "1.0": ("tabulartoy_25_10k_models_acem_shared_critic", "CEM_adam_lr0.05_bs64_lam_c1"),
        },
        "GC-CEM": {
            "0.1": ("tabulartoy_25_10k_models_acem_shared_critic", "ACEM_adam_lr1e-03_bs64_lam1_none_lam_c0.1_shared_critic"),
            "0.5": ("tabulartoy_25_10k_models_acem_shared_critic", "ACEM_adam_lr1e-03_bs64_lam1_none_lam_c0.5_shared_critic"),
            "1.0": ("tabulartoy_25_10k_models_acem_shared_critic", "ACEM_adam_lr1e-03_bs64_lam1_none_lam_c1_shared_critic"),
        },
    },
    "dS": {
        "CEM": {
            "0.1": ("dsprites_cem", "CEM_adam_lr1e-03_bs64_lam_c0.1"),
            "0.5": ("dsprites_cem", "CEM_adam_lr1e-03_bs64_lam_c0.5"),
            "1.0": ("dsprites_cem", "CEM_adam_lr1e-03_bs64_lam_c1"),
        },
        "GC-CEM": {
            "0.1": ("dsprites_acem_shared_critic", "CRCEM_adam_lr5e-05_bs64_lam1_none_lam_c0.1_shared_critic"),
            "0.5": ("dsprites_acem_shared_critic", "CRCEM_adam_lr5e-05_bs64_lam1_none_lam_c0.5_shared_critic"),
            "1.0": ("dsprites_acem_shared_critic", "CRCEM_adam_lr5e-05_bs64_lam1_none_lam_c1_shared_critic"),
        },
    },
    "CUB": {
        "CEM": {
            "0.1": ("cub_cem", "CEM_adam_lr1e-03_bs256_lam_c0.1"),
            "0.5": ("cub_cem", "CEM_adam_lr1e-03_bs256_lam_c0.5"),
            "1.0": ("cub_cem", "CEM_adam_lr1e-03_bs256_lam_c1"),
        },
        "GC-CEM": {
            "0.1": ("cub_cem", "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c0.1_shared_critic"),
            "0.5": ("cub_cem", "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c0.5_shared_critic"),
            "1.0": ("cub_cem", "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c1_shared_critic"),
        },
    },
}

PAPER_CEM_ACC = {
    "TT": {
        "CEM 0.1":    (99.6, 64.1), "CEM 0.5":    (99.5, 98.8), "CEM 1.0":    (99.7, 99.5),
        "GC-CEM 0.1": (99.6, 99.2), "GC-CEM 0.5": (99.5, 99.2), "GC-CEM 1.0": (99.5, 99.2),
    },
    "dS": {
        "CEM 0.1":    (91.6, 78.0), "CEM 0.5":    (96.6, 96.3), "CEM 1.0":    (95.6, 95.3),
        "GC-CEM 0.1": (96.2, 96.3), "GC-CEM 0.5": (96.7, 96.7), "GC-CEM 1.0": (96.3, 96.3),
    },
    "CUB": {
        "CEM 0.1":    (72.1, 91.1), "CEM 0.5":    (73.2, 93.8), "CEM 1.0":    (73.9, 95.1),
        "GC-CEM 0.1": (69.1, 95.1), "GC-CEM 0.5": (70.6, 95.4), "GC-CEM 1.0": (70.1, 95.3),
    },
}

PAPER_CEM_CTL = {
    "TT": {
        "CEM 0.1":    (0.574, 0.365), "CEM 0.5":    (0.472, 0.247), "CEM 1.0":    (0.350, 0.209),
        "GC-CEM 0.1": (0.083, 0.036), "GC-CEM 0.5": (0.055, 0.022), "GC-CEM 1.0": (0.053, 0.019),
    },
    "dS": {
        "CEM 0.1":    (0.251, 0.105), "CEM 0.5":    (0.131, 0.070), "CEM 1.0":    (0.111, 0.075),
        "GC-CEM 0.1": (0.015, 0.025), "GC-CEM 0.5": (0.003, 0.013), "GC-CEM 1.0": (0.003, 0.012),
    },
    "CUB": {
        "CEM 0.1":    (0.032, 0.001), "CEM 0.5":    (0.035, 0.002), "CEM 1.0":    (0.029, 0.003),
        "GC-CEM 0.1": (0.039, 0.005), "GC-CEM 0.5": (0.031, 0.006), "GC-CEM 1.0": (0.036, 0.004),
    },
}

LAM_MAP = {"0.1": "lam_c0.1", "0.5": "lam_c0.5", "1.0": "lam_c1"}

CEM_SUITE = {
    "TT": {"CEM": (tt_cem, "cem"),  "GC-CEM": (tt_cem, "acem")},
    "dS": {"CEM": (ds_cem, "cem"),  "GC-CEM": (ds_cem, "crcem")},
    "CUB":{"CEM": (cub_cem, "cem"), "GC-CEM": (cub_cem, "crcem")},
}

for ds_label in ["TT", "dS", "CUB"]:
    for model_label in ["CEM", "GC-CEM"]:
        suite_d, suite_key = CEM_SUITE[ds_label][model_label]
        for lam in ["0.1", "0.5", "1.0"]:
            paper_key = f"{model_label} {lam}"
            lam_key   = LAM_MAP[lam]
            folder, mname = CEM_SPLITS[ds_label][model_label][lam]
            folds = load_splits(folder, mname)

            if not folds:
                task_s = c_s = ctl_s = icl_s = "MISSING"
            else:
                if ds_label == "CUB":
                    # CUB: suite dict has only lam_c0.1 and wrong c_acc (joint of 112 → ~0)
                    # Paper reports per-concept c_acc; read from split results
                    task_s = s(get(folds, "test_acc_y", 100))
                    c_s    = s(get(folds, "test_acc_c", 100))
                else:
                    # TT / dS: paper reports joint concept accuracy; suite dict is correct
                    folds_s = suite_d.get(suite_key, {}).get(lam_key, {})
                    task_vs = [float(folds_s.get(fk, {}).get("task_acc", np.nan)) * 100
                               for fk in folds_s]
                    c_vs    = [float(folds_s.get(fk, {}).get("c_acc", np.nan)) * 100
                               for fk in folds_s]
                    task_s = s(task_vs) if task_vs else "MISSING"
                    c_s    = s(c_vs)    if c_vs    else "MISSING"
                ctl_s  = s(ctl_vals(folds))
                icl_s  = s(icl_vals(folds))

            pt, pc   = PAPER_CEM_ACC[ds_label].get(paper_key, (np.nan, np.nan))
            pctl, picl = PAPER_CEM_CTL[ds_label].get(paper_key, (np.nan, np.nan))
            print(f"{model_label:<12} {lam:<5} {ds_label:<4} | {task_s:>12} {c_s:>12} {ctl_s:>12} {icl_s:>12} | {pt:>7.1f} {pc:>6.1f} {pctl:>7.3f} {picl:>7.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 5: CVL / ICVL
# Sources: TT → acem_suite dict (cvl/icvl); dS → ds_cvl_onehot dict (cvl_clipped_mean) + ds_suite (icvl)
#          CUB → cub_suite dict (all 0.00 after clipping)
# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("TAB 5: CVL / ICVL")
print(f"{'Model':<12} {'λ':<5} {'DS':<4} | {'CVL':>12} {'ICVL':>12} | {'p_CVL':>7} {'p_ICVL':>7}")
print(HSEP)

PAPER_CVL = {
    "TT": {
        "CEM 0.1":    (0.16, 0.21), "CEM 0.5":    (0.12, 0.20), "CEM 1.0":    (0.12, 0.22),
        "GC-CEM 0.1": (0.02, 0.04), "GC-CEM 0.5": (0.02, 0.04), "GC-CEM 1.0": (0.01, 0.03),
    },
    "dS": {
        "CEM 0.1":    (0.56, 0.06), "CEM 0.5":    (0.53, 0.06), "CEM 1.0":    (0.60, 0.07),
        "GC-CEM 0.1": (0.12, 0.02), "GC-CEM 0.5": (0.09, 0.01), "GC-CEM 1.0": (0.08, 0.01),
    },
    "CUB": {
        "CEM 0.1":    (0.00, 0.00), "CEM 0.5":    (0.00, 0.00), "CEM 1.0":    (0.00, 0.00),
        "GC-CEM 0.1": (0.00, 0.00), "GC-CEM 0.5": (0.00, 0.00), "GC-CEM 1.0": (0.00, 0.00),
    },
}

LAM_MAP = {"0.1": "lam_c0.1", "0.5": "lam_c0.5", "1.0": "lam_c1"}
SUITE_KEYS = {
    "TT":  {"CEM": (tt_cem, "cem"),    "GC-CEM": (tt_cem, "acem")},
    "dS":  {"CEM": (ds_cem, "cem"),    "GC-CEM": (ds_cem, "crcem")},
    "CUB": {"CEM": (cub_cem, "cem"),   "GC-CEM": (cub_cem, "crcem")},
}
ONEHOT_KEYS = {
    "TT":  {"CEM": "cem", "GC-CEM": "gc_cem"},
    "dS":  {"CEM": "cem", "GC-CEM": "gc_cem"},
}

for ds_label in ["TT", "dS", "CUB"]:
    for model_label in ["CEM", "GC-CEM"]:
        suite_d, suite_key = SUITE_KEYS[ds_label][model_label]
        for lam in ["0.1", "0.5", "1.0"]:
            lam_key   = LAM_MAP[lam]
            paper_key = f"{model_label} {lam}"

            if ds_label == "CUB":
                # CUB: all clipped to 0.00; read from suite dict if available
                folds_s = suite_d.get(suite_key, {}).get(lam_key, {})
                if folds_s:
                    cvl_vs  = [float(folds_s.get(fk, {}).get("cvl", {}).get("CVL", 0.0))
                               for fk in folds_s]
                    icvl_vs = [float(folds_s.get(fk, {}).get("icvl", {}).get("ICVL", 0.0))
                               for fk in folds_s]
                    # clip to 0 (negative R² → 0)
                    cvl_vs  = [max(0.0, v) for v in cvl_vs]
                    icvl_vs = [max(0.0, v) for v in icvl_vs]
                    cvl_s  = s(cvl_vs)
                    icvl_s = s(icvl_vs)
                else:
                    cvl_s = icvl_s = "0.000±0.000"
            elif ds_label == "TT":
                # TT: CVL and ICVL from acem_suite dict (cvl['CVL'], icvl['ICVL'])
                folds_s = suite_d.get(suite_key, {}).get(lam_key, {})
                if not folds_s:
                    cvl_s = icvl_s = "MISSING"
                else:
                    cvl_vs  = [float(folds_s.get(fk, {}).get("cvl",  {}).get("CVL",  np.nan))
                               for fk in folds_s]
                    icvl_vs = [float(folds_s.get(fk, {}).get("icvl", {}).get("ICVL", np.nan))
                               for fk in folds_s]
                    cvl_s  = s(cvl_vs)
                    icvl_s = s(icvl_vs)
            else:
                # dS: CVL from onehot dict (cvl_clipped_mean), ICVL from suite dict
                oh_key = ONEHOT_KEYS[ds_label][model_label]
                oh_folds = ds_onehot.get(oh_key, {}).get(lam_key, {})
                if oh_folds:
                    cvl_vs = [float(fd.get("cvl_clipped_mean", np.nan))
                              for fd in oh_folds.values()]
                    cvl_s = s(cvl_vs)
                else:
                    cvl_s = "MISSING"
                folds_s = suite_d.get(suite_key, {}).get(lam_key, {})
                if folds_s:
                    icvl_vs = [float(folds_s.get(fk, {}).get("icvl", {}).get("ICVL", np.nan))
                               for fk in folds_s]
                    icvl_s = s(icvl_vs)
                else:
                    icvl_s = "MISSING"

            pcvl, picvl = PAPER_CVL[ds_label].get(paper_key, (np.nan, np.nan))
            print(f"{model_label:<12} {lam:<5} {ds_label:<4} | {cvl_s:>12} {icvl_s:>12} | {pcvl:>7.2f} {picvl:>7.2f}")
