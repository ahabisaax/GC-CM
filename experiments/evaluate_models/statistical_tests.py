"""
Welch's t-tests comparing GC models vs baselines across all metrics and datasets.

Comparisons:
  1. GC-CBM vs Joint CBM  (matched lambda)
  2. GC-CBM vs Sequential CBM
  3. GC-CBM vs Hard CBM
  4. GC-CEM vs CEM        (matched lambda)

Metrics: task accuracy, concept accuracy, CTL, ICL, CVL (CEM only), ICVL (CEM only)

Significance: ns (p>=0.05), * (p<0.05), ** (p<0.01), *** (p<0.001)

Run from project root:
    python experiments/evaluate_models/statistical_tests.py
"""
import os
import sys
import warnings
import numpy as np
import joblib
from scipy import stats

warnings.filterwarnings("ignore")

BASE = os.getcwd() + "/results/"
N_SPLITS = 5


# ── helpers ──────────────────────────────────────────────────────────────────

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def cohens_d(a, b):
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / pooled_std) if pooled_std > 0 else 0.0


def welch_test(a, b):
    a = [x for x in a if not np.isnan(x)]
    b = [x for x in b if not np.isnan(x)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def fmt(vals):
    v = [x for x in vals if not np.isnan(x)]
    if not v:
        return "    —"
    return f"{np.mean(v):.3f}±{np.std(v):.3f}"


def load_splits(folder, model_name, n=5):
    folds = []
    alt = model_name.replace("lr1e-04", "lr1e-4")
    if alt == model_name:
        alt = model_name.replace("lr1e-4", "lr1e-04")
    for i in range(n):
        candidates = [
            os.path.join(BASE, folder, model_name, f"{model_name}_split_{i}_results.joblib"),
            os.path.join(BASE, folder, model_name, f"{alt}_split_{i}_results.joblib"),
            os.path.join(BASE, folder, alt,        f"{alt}_split_{i}_results.joblib"),
        ]
        for p in candidates:
            if os.path.exists(p):
                folds.append(joblib.load(p))
                break
    return folds


def ctl_from_splits(folds):
    raw = [float(f.get("test_ctl_average") or np.nan) for f in folds]
    if not all(np.isnan(x) for x in raw):
        return raw
    return [float(f.get("test_normalised_ctl") or np.nan) for f in folds]


def icl_from_splits(folds):
    avg = [float(f.get("test_icl_average") or np.nan) for f in folds]
    avg_mean = np.nanmean(avg)
    if not np.isnan(avg_mean) and avg_mean <= 1.0:
        return avg
    return [float(f.get("test_normalised_icl") or np.nan) for f in folds]


def acc_y(folds):   return [float(f.get("test_acc_y", np.nan)) * 100 for f in folds]
def acc_c(folds):   return [float(f.get("test_acc_c", np.nan)) * 100 for f in folds]


def suite_metric(suite, model_key, lam, metric_fn):
    folds = suite.get(model_key, {}).get(lam, {})
    return [metric_fn(folds.get(f, {})) for f in ["fold_1","fold_2","fold_3","fold_4","fold_5"]]


def _cvl(d):  return float(d.get("cvl",  {}).get("CVL",  np.nan))
def _icvl(d): return float(d.get("icvl", {}).get("ICVL", np.nan))
def _task(d): return float(d.get("task_acc", np.nan)) * 100
def _cacc(d): return float(d.get("c_acc",    np.nan)) * 100
def _ctl(d):  return float(d.get("ctl_mean", [np.nan])[0]) if isinstance(d.get("ctl_mean"), list) else float(d.get("ctl_mean", np.nan))
def _icl(d):  return float(d.get("icl_mean", np.nan))


SEP  = "=" * 110
HSEP = "-" * 110

def print_row(label, a_vals, b_vals, header=False):
    if header:
        print(f"  {'Metric':<18} {'GC':>12} {'Base':>12} {'t':>7} {'df':>5} {'p':>8} {'sig':>4} {'d':>6}  {'direction'}")
        print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*7} {'-'*5} {'-'*8} {'-'*4} {'-'*6}  {'-'*20}")
        return
    a = [x for x in a_vals if not np.isnan(x)]
    b = [x for x in b_vals if not np.isnan(x)]
    if not a or not b:
        print(f"  {label:<18}  MISSING")
        return
    t, p = welch_test(a, b)
    if np.isnan(p):
        print(f"  {label:<18}  n/a")
        return
    df = (np.var(a, ddof=1)/len(a) + np.var(b, ddof=1)/len(b))**2 / (
         (np.var(a, ddof=1)/len(a))**2/(len(a)-1) + (np.var(b, ddof=1)/len(b))**2/(len(b)-1))
    d  = cohens_d(a, b)
    direction = "GC lower ✓" if np.mean(a) < np.mean(b) else ("GC higher ✓" if label in ("task%", "c_acc%") else "GC higher ✗")
    print(f"  {label:<18} {fmt(a):>12} {fmt(b):>12} {t:>7.2f} {df:>5.1f} {p:>8.4f} {stars(p):>4} {d:>6.2f}  {direction}")


# ── load data ────────────────────────────────────────────────────────────────

tt_cem  = joblib.load(BASE + "results_tabulartoy_acem_suite.dict")
tt_cbm  = joblib.load(BASE + "results_tabulartoy_cbm_suite.dict")
ds_cem  = joblib.load(BASE + "results_dsprites_suite.dict")
ds_cbm  = joblib.load(BASE + "results_dsprites_cbm_suite.dict")

# onehot CVL dicts (for dSprites CVL; rename to cvl if new script has run)
ds_cvl_path = BASE + "results_dsprites_cvl.dict"
if not os.path.exists(ds_cvl_path):
    ds_cvl_path = BASE + "results_dsprites_cvl_onehot.dict"
ds_cvl = joblib.load(ds_cvl_path) if os.path.exists(ds_cvl_path) else {}

tt_cvl_path = BASE + "results_tabulartoy_cvl.dict"
if not os.path.exists(tt_cvl_path):
    tt_cvl_path = BASE + "results_tabulartoy_cvl_onehot.dict"
tt_cvl = joblib.load(tt_cvl_path) if os.path.exists(tt_cvl_path) else {}

LAMS = ["0.1", "0.5", "1.0"]
LAM_MAP = {"0.1": "lam_c0.1", "0.5": "lam_c0.5", "1.0": "lam_c1"}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: GC-CBM vs Joint CBM (TabularToy)
# ═════════════════════════════════════════════════════════════════════════════
print(SEP)
print("GC-CBM vs JOINT CBM — TabularToy (per λ)")
print(SEP)
print_row(None, [], [], header=True)

for lam in LAMS:
    lk = LAM_MAP[lam]
    gc_splits = load_splits("tabulartoy_25_10k_models_acbm_shared_critic",
                            f"ACBM_adam_lr0.05_bs64_lam1_none_lam_c{lam}_shared_critic")
    jt_splits = load_splits("tabulartoy_25_10k_models",
                            f"SoftCBM_adam_lr0.05_bs64_lam_c{lam}")
    if not gc_splits or not jt_splits:
        print(f"\n  λ={lam}: missing splits"); continue
    print(f"\n  λ_c = {lam}")
    print_row("task%",   acc_y(gc_splits),    acc_y(jt_splits))
    print_row("c_acc%",  acc_c(gc_splits),    acc_c(jt_splits))
    print_row("CTL",     ctl_from_splits(gc_splits), ctl_from_splits(jt_splits))
    print_row("ICL",     icl_from_splits(gc_splits), icl_from_splits(jt_splits))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: GC-CBM vs Sequential CBM (TabularToy)
# ═════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("GC-CBM vs SEQUENTIAL CBM — TabularToy")
print(SEP)
print_row(None, [], [], header=True)

seq_splits = load_splits("tabulartoy_25_10k_models_acbm_shared_critic",
                         "SeqCBM_adam_lr0.05_bs64_lam_c1")
if not seq_splits:
    seq_splits = load_splits("tabulartoy_25_10k_models_sequential_cbm",
                             "SeqCBM_adam_lr0.05_bs64_lam_c1")

for lam in LAMS:
    gc_splits = load_splits("tabulartoy_25_10k_models_acbm_shared_critic",
                            f"ACBM_adam_lr0.05_bs64_lam1_none_lam_c{lam}_shared_critic")
    if not gc_splits or not seq_splits:
        continue
    print(f"\n  GC-CBM λ_c={lam} vs Seq CBM")
    print_row("task%",  acc_y(gc_splits),  acc_y(seq_splits))
    print_row("c_acc%", acc_c(gc_splits),  acc_c(seq_splits))
    print_row("CTL",    ctl_from_splits(gc_splits), ctl_from_splits(seq_splits))
    print_row("ICL",    icl_from_splits(gc_splits), icl_from_splits(seq_splits))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: GC-CBM vs Hard CBM (TabularToy)
# ═════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("GC-CBM vs HARD CBM — TabularToy")
print(SEP)
print_row(None, [], [], header=True)

hard_splits = load_splits("tabulartoy_25_10k_models",
                          "HardCBM_adam_lr0.05_bs64_lam_c1")

for lam in LAMS:
    gc_splits = load_splits("tabulartoy_25_10k_models_acbm_shared_critic",
                            f"ACBM_adam_lr0.05_bs64_lam1_none_lam_c{lam}_shared_critic")
    if not gc_splits or not hard_splits:
        continue
    print(f"\n  GC-CBM λ_c={lam} vs Hard CBM")
    print_row("task%",  acc_y(gc_splits),  acc_y(hard_splits))
    print_row("c_acc%", acc_c(gc_splits),  acc_c(hard_splits))
    print_row("CTL",    ctl_from_splits(gc_splits), ctl_from_splits(hard_splits))
    print_row("ICL",    icl_from_splits(gc_splits), icl_from_splits(hard_splits))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: GC-CEM vs CEM — TabularToy (split results + suite dict)
# ═════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("GC-CEM vs CEM — TabularToy")
print(SEP)
print_row(None, [], [], header=True)

for lam in LAMS:
    lk = LAM_MAP[lam]
    gc_splits = load_splits("tabulartoy_25_10k_models_acem_shared_critic",
                            f"ACEM_adam_lr1e-03_bs64_lam1_none_lam_c{lam}_shared_critic")
    ce_splits = load_splits("tabulartoy_25_10k_models_acem_shared_critic",
                            f"CEM_adam_lr0.05_bs64_lam_c{lam}")

    gc_suite = suite_metric(tt_cem, "acem", lk, lambda d: d)
    ce_suite = suite_metric(tt_cem, "cem",  lk, lambda d: d)

    # CVL from onehot/cvl dict
    gc_cvl = [float(tt_cvl.get("gc_cem", {}).get(lk, {}).get(f, {}).get("cvl_clipped_mean", np.nan))
              for f in ["fold_1","fold_2","fold_3","fold_4","fold_5"]]
    ce_cvl = [float(tt_cvl.get("cem",    {}).get(lk, {}).get(f, {}).get("cvl_clipped_mean", np.nan))
              for f in ["fold_1","fold_2","fold_3","fold_4","fold_5"]]

    if not gc_splits and not any(np.isfinite(x) for x in [_task(d) for d in gc_suite if d]):
        print(f"\n  λ_c={lam}: missing data"); continue

    print(f"\n  λ_c = {lam}")
    if gc_splits and ce_splits:
        print_row("task%",  acc_y(gc_splits),  acc_y(ce_splits))
        print_row("c_acc%", acc_c(gc_splits),  acc_c(ce_splits))
        print_row("CTL",    ctl_from_splits(gc_splits), ctl_from_splits(ce_splits))
        print_row("ICL",    icl_from_splits(gc_splits), icl_from_splits(ce_splits))
    elif gc_suite and ce_suite:
        gc_task  = [_task(d) for d in gc_suite if d]
        ce_task  = [_task(d) for d in ce_suite if d]
        gc_cacc  = [_cacc(d) for d in gc_suite if d]
        ce_cacc  = [_cacc(d) for d in ce_suite if d]
        print_row("task%",  gc_task, ce_task)
        print_row("c_acc%", gc_cacc, ce_cacc)

    # CVL / ICVL from suite dict
    gc_cvl_suite  = [_cvl(d)  for d in gc_suite if d]
    ce_cvl_suite  = [_cvl(d)  for d in ce_suite if d]
    gc_icvl_suite = [_icvl(d) for d in gc_suite if d]
    ce_icvl_suite = [_icvl(d) for d in ce_suite if d]

    # prefer onehot/cvl dict values for CVL if available
    use_gc_cvl = gc_cvl if any(np.isfinite(x) for x in gc_cvl) else gc_cvl_suite
    use_ce_cvl = ce_cvl if any(np.isfinite(x) for x in ce_cvl) else ce_cvl_suite

    print_row("CVL",  use_gc_cvl,    use_ce_cvl)
    print_row("ICVL", gc_icvl_suite, ce_icvl_suite)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: GC-CBM vs Joint CBM — dSprites
# ═════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("GC-CBM vs JOINT CBM — dSprites (per λ)")
print(SEP)
print_row(None, [], [], header=True)

for lam in LAMS:
    lk = LAM_MAP[lam]
    gc_splits = load_splits("dsprites_ACBM_shared_critic",
                            f"CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c{lam}_shared_critic")
    if not gc_splits:
        gc_splits = load_splits("dsprites_ACBM_shared_critic",
                                f"CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c{lam}_shared_critic".replace("lr5e-05","lr5e-5"))
    jt_splits = load_splits("dsprites_dep_0_models",
                            f"SoftCBM_adam_lr1e-04_bs64_lam_c{lam}")
    if not gc_splits or not jt_splits:
        print(f"\n  λ={lam}: missing"); continue
    print(f"\n  λ_c = {lam}")
    print_row("task%",  acc_y(gc_splits),  acc_y(jt_splits))
    print_row("c_acc%", acc_c(gc_splits),  acc_c(jt_splits))
    print_row("CTL",    ctl_from_splits(gc_splits), ctl_from_splits(jt_splits))
    print_row("ICL",    icl_from_splits(gc_splits), icl_from_splits(jt_splits))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: GC-CBM vs Sequential CBM — dSprites
# ═════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("GC-CBM vs SEQUENTIAL CBM — dSprites")
print(SEP)
print_row(None, [], [], header=True)

seq_ds_splits = load_splits("dsprites_sequential_acbm", "SeqCBM_adam_lr1e-04_bs64_lam_c1")

for lam in LAMS:
    gc_splits = load_splits("dsprites_ACBM_shared_critic",
                            f"CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c{lam}_shared_critic")
    if not gc_splits or not seq_ds_splits:
        continue
    print(f"\n  GC-CBM λ_c={lam} vs Seq CBM")
    print_row("task%",  acc_y(gc_splits), acc_y(seq_ds_splits))
    print_row("c_acc%", acc_c(gc_splits), acc_c(seq_ds_splits))
    print_row("CTL",    ctl_from_splits(gc_splits), ctl_from_splits(seq_ds_splits))
    print_row("ICL",    icl_from_splits(gc_splits), icl_from_splits(seq_ds_splits))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: GC-CBM vs Hard CBM — dSprites
# ═════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("GC-CBM vs HARD CBM — dSprites")
print(SEP)
print_row(None, [], [], header=True)

hard_ds_splits = load_splits("dsprites_dep_0_models", "HardCBM_adam_lr1e-04_bs64_lam_c1")

for lam in LAMS:
    gc_splits = load_splits("dsprites_ACBM_shared_critic",
                            f"CRCBM_adam_lr5e-05_bs64_lam1_none_lam_c{lam}_shared_critic")
    if not gc_splits or not hard_ds_splits:
        continue
    print(f"\n  GC-CBM λ_c={lam} vs Hard CBM")
    print_row("task%",  acc_y(gc_splits), acc_y(hard_ds_splits))
    print_row("c_acc%", acc_c(gc_splits), acc_c(hard_ds_splits))
    print_row("CTL",    ctl_from_splits(gc_splits), ctl_from_splits(hard_ds_splits))
    print_row("ICL",    icl_from_splits(gc_splits), icl_from_splits(hard_ds_splits))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: GC-CEM vs CEM — dSprites
# ═════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("GC-CEM vs CEM — dSprites")
print(SEP)
print_row(None, [], [], header=True)

for lam in LAMS:
    lk = LAM_MAP[lam]
    gc_splits = load_splits("dsprites_acem_shared_critic",
                            f"CRCEM_adam_lr5e-05_bs64_lam1_none_lam_c{lam}_shared_critic")
    ce_splits = load_splits("dsprites_cem", f"CEM_adam_lr1e-03_bs64_lam_c{lam}")

    gc_suite = [ds_cem.get("crcem", {}).get(lk, {}).get(f, {})
                for f in ["fold_1","fold_2","fold_3","fold_4","fold_5"]]
    ce_suite = [ds_cem.get("cem",   {}).get(lk, {}).get(f, {})
                for f in ["fold_1","fold_2","fold_3","fold_4","fold_5"]]

    # CVL from cvl dict
    gc_cvl = [float(ds_cvl.get("gc_cem", {}).get(lk, {}).get(f, {}).get("cvl_clipped_mean", np.nan))
              for f in ["fold_1","fold_2","fold_3","fold_4","fold_5"]]
    ce_cvl = [float(ds_cvl.get("cem",    {}).get(lk, {}).get(f, {}).get("cvl_clipped_mean", np.nan))
              for f in ["fold_1","fold_2","fold_3","fold_4","fold_5"]]

    print(f"\n  λ_c = {lam}")
    if gc_splits and ce_splits:
        print_row("task%",  acc_y(gc_splits),  acc_y(ce_splits))
        print_row("c_acc%", acc_c(gc_splits),  acc_c(ce_splits))
        print_row("CTL",    ctl_from_splits(gc_splits), ctl_from_splits(ce_splits))
        print_row("ICL",    icl_from_splits(gc_splits), icl_from_splits(ce_splits))
    else:
        gc_task = [_task(d) for d in gc_suite if d]
        ce_task = [_task(d) for d in ce_suite if d]
        print_row("task%", gc_task, ce_task)

    gc_cvl_suite  = [_cvl(d)  for d in gc_suite if d]
    ce_cvl_suite  = [_cvl(d)  for d in ce_suite if d]
    gc_icvl_suite = [_icvl(d) for d in gc_suite if d]
    ce_icvl_suite = [_icvl(d) for d in ce_suite if d]

    use_gc_cvl = gc_cvl if any(np.isfinite(x) for x in gc_cvl) else gc_cvl_suite
    use_ce_cvl = ce_cvl if any(np.isfinite(x) for x in ce_cvl) else ce_cvl_suite

    print_row("CVL",  use_gc_cvl,    use_ce_cvl)
    print_row("ICVL", gc_icvl_suite, ce_icvl_suite)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9: GC-CBM vs Joint CBM — CUB
# ═════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("GC-CBM vs JOINT CBM — CUB-200 (per λ)")
print(SEP)
print_row(None, [], [], header=True)

for lam in LAMS:
    gc_splits = load_splits("cub_acbm_shared_critic",
                            f"ACBM_adam_lr1e-04_bs256_lam1_none_lam_c{lam}_shared_critic")
    jt_splits = load_splits("cub_soft_0.01",
                            f"SoftCBM_adam_lr1e-04_bs256_lam_c{lam}")
    if not gc_splits or not jt_splits:
        print(f"\n  λ={lam}: missing splits"); continue
    print(f"\n  λ_c = {lam}")
    print_row("task%",  acc_y(gc_splits), acc_y(jt_splits))
    print_row("c_acc%", acc_c(gc_splits), acc_c(jt_splits))
    print_row("CTL",    ctl_from_splits(gc_splits), ctl_from_splits(jt_splits))
    print_row("ICL",    icl_from_splits(gc_splits), icl_from_splits(jt_splits))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10: GC-CBM vs Sequential CBM — CUB
# ═════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("GC-CBM vs SEQUENTIAL CBM — CUB-200")
print(SEP)
print_row(None, [], [], header=True)

seq_cub_splits = load_splits("cub_acbm_shared_critic", "SeqCBM_adam_lr1e-04_bs256_lam_c1")

for lam in LAMS:
    gc_splits = load_splits("cub_acbm_shared_critic",
                            f"ACBM_adam_lr1e-04_bs256_lam1_none_lam_c{lam}_shared_critic")
    if not gc_splits or not seq_cub_splits:
        continue
    print(f"\n  GC-CBM λ_c={lam} vs Seq CBM")
    print_row("task%",  acc_y(gc_splits), acc_y(seq_cub_splits))
    print_row("c_acc%", acc_c(gc_splits), acc_c(seq_cub_splits))
    print_row("CTL",    ctl_from_splits(gc_splits), ctl_from_splits(seq_cub_splits))
    print_row("ICL",    icl_from_splits(gc_splits), icl_from_splits(seq_cub_splits))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11: GC-CBM vs Hard CBM — CUB
# ═════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("GC-CBM vs HARD CBM — CUB-200")
print(SEP)
print_row(None, [], [], header=True)

hard_cub_splits = load_splits("cub_soft_0.01", "HardCBM_adam_lr1e-04_bs256_lam_c1")

for lam in LAMS:
    gc_splits = load_splits("cub_acbm_shared_critic",
                            f"ACBM_adam_lr1e-04_bs256_lam1_none_lam_c{lam}_shared_critic")
    if not gc_splits or not hard_cub_splits:
        continue
    print(f"\n  GC-CBM λ_c={lam} vs Hard CBM")
    print_row("task%",  acc_y(gc_splits), acc_y(hard_cub_splits))
    print_row("c_acc%", acc_c(gc_splits), acc_c(hard_cub_splits))
    print_row("CTL",    ctl_from_splits(gc_splits), ctl_from_splits(hard_cub_splits))
    print_row("ICL",    icl_from_splits(gc_splits), icl_from_splits(hard_cub_splits))


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 12: GC-CEM vs CEM — CUB
# ═════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("GC-CEM vs CEM — CUB-200")
print(SEP)
print_row(None, [], [], header=True)

cub_suite_path = BASE + "results_cub_suite.dict"
cub_cem_suite  = joblib.load(cub_suite_path) if os.path.exists(cub_suite_path) else {}

for lam in LAMS:
    lk = LAM_MAP[lam]
    gc_splits = load_splits("cub_cem",
                            f"CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c{lam}_shared_critic")
    ce_splits = load_splits("cub_cem", f"CEM_adam_lr1e-03_bs256_lam_c{lam}")

    gc_suite = [cub_cem_suite.get("crcem", {}).get(lk, {}).get(f, {})
                for f in ["fold_1","fold_2","fold_3","fold_4","fold_5"]]
    ce_suite = [cub_cem_suite.get("cem",   {}).get(lk, {}).get(f, {})
                for f in ["fold_1","fold_2","fold_3","fold_4","fold_5"]]

    print(f"\n  λ_c = {lam}")
    if gc_splits and ce_splits:
        print_row("task%",  acc_y(gc_splits), acc_y(ce_splits))
        print_row("c_acc%", acc_c(gc_splits), acc_c(ce_splits))
        print_row("CTL",    ctl_from_splits(gc_splits), ctl_from_splits(ce_splits))
        print_row("ICL",    icl_from_splits(gc_splits), icl_from_splits(ce_splits))
    elif any(gc_suite) and any(ce_suite):
        gc_task = [_task(d) for d in gc_suite if d]
        ce_task = [_task(d) for d in ce_suite if d]
        print_row("task%", gc_task, ce_task)

    # CVL/ICVL are 0.00 for all CUB CEMs (negative R² clipped to 0)
    gc_cvl_suite  = [_cvl(d)  for d in gc_suite if d]
    ce_cvl_suite  = [_cvl(d)  for d in ce_suite if d]
    gc_icvl_suite = [_icvl(d) for d in gc_suite if d]
    ce_icvl_suite = [_icvl(d) for d in ce_suite if d]
    print_row("CVL",  gc_cvl_suite,  ce_cvl_suite)
    print_row("ICVL", gc_icvl_suite, ce_icvl_suite)


print()
print(SEP)
print("KEY: GC column = GC model; Base column = baseline model")
print("     d = Cohen's d (GC − Baseline); negative d means GC has lower value")
print("     For leakage metrics (CTL, ICL, CVL, ICVL): lower is better → expect d < 0 and sig")
print("     For accuracy metrics (task%, c_acc%): higher is better → ns expected (no sig. loss)")
print(SEP)
