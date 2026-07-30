"""
Comprehensive t-tests across all datasets, model pairs, lambdas, and metrics.
Outputs a table flagging which comparisons are significant (p<0.05).

Run:
    conda run -n xai-leakage python experiments/evaluate_models/ttest_all.py
"""
import joblib
import numpy as np
from scipy import stats

SEP = "-" * 90


# ── helpers ──────────────────────────────────────────────────────────────────

def scalar(v, key=None):
    """Extract a scalar from a metric value (float, list, or dict)."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list):
        if not v:
            return None
        # ctl_mean: list of 3 MI estimates → mean
        if isinstance(v[0], (int, float)):
            return float(np.mean(v))
        # probe_acc: list of repeats, each a list of accs per probe size → mean of final acc
        if isinstance(v[0], list):
            return float(np.mean([r[-1] for r in v if r]))
    if isinstance(v, dict):
        # icl_probe: pairwise matrix — mean off-diagonal acc_diff
        if key == "icl_probe" and "acc_diff" in v:
            m = np.array(v["acc_diff"])
            off = m[~np.eye(len(m), dtype=bool)]
            return float(off.mean())
        # ds_leak: leakage rate
        if "leakage_rate" in v:
            return float(v["leakage_rate"])
        # emb_probe_delta: delta = acc_emb - acc_scalar
        if "delta" in v:
            return float(v["delta"])
        # task_leak: acc_delta = acc_concat - acc_concepts
        if "acc_delta" in v:
            return float(v["acc_delta"])
        for k in ("CVL", "ICVL", "CVL_global", "task_acc", "c_acc",
                  "mean", "ICVL_mean"):
            if k in v:
                return float(v[k])
    return None


def get_folds(r, model, metric, lam=None):
    """Return per-fold scalar values for a given model/lam/metric."""
    folds = r[model] if lam is None else r[model].get(lam, {})
    out = []
    for fold_data in folds.values():
        if not isinstance(fold_data, dict):
            continue
        v = fold_data.get(metric)
        if v is None:
            continue
        s = scalar(v, key=metric)
        if s is not None:
            out.append(s)
    return out


def ttest(a_vals, b_vals):
    """Welch t-test; returns (t, p) or (nan, nan) if insufficient data or zero variance."""
    a, b = np.array(a_vals), np.array(b_vals)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    if np.std(a) == 0 and np.std(b) == 0:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def row(metric, a_vals, b_vals, a_name, b_name, scale=1.0, fmt=".3f"):
    """Print one comparison row."""
    a, b = np.array(a_vals) * scale, np.array(b_vals) * scale
    t, p = ttest(a_vals, b_vals)  # t-test on unscaled; p same either way
    if np.isnan(p):
        sig = "-- (undefined)"
    elif p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "** "
    elif p < 0.05:
        sig = "*  "
    else:
        sig = "ns "
    bold_a = "<" if np.mean(a) > np.mean(b) else " "
    bold_b = "<" if np.mean(b) > np.mean(a) else " "
    # for leakage metrics, lower is better → flip bold arrows
    print(
        f"  {metric:18s}  "
        f"{a_name:>10} {np.mean(a):{fmt}}±{np.std(a):{fmt}}  "
        f"{b_name:>10} {np.mean(b):{fmt}}±{np.std(b):{fmt}}  "
        f"p={p:.3f} {sig}  n={len(a_vals)}/{len(b_vals)}"
    )


def section(title):
    print()
    print(SEP)
    print(title)
    print(SEP)


def compare(r, model_a, model_b, metrics, lam=None, scale=1.0, fmt=".3f",
            a_label=None, b_label=None):
    a_lbl = a_label or model_a
    b_lbl = b_label or model_b
    for metric in metrics:
        a_vals = get_folds(r, model_a, metric, lam)
        b_vals = get_folds(r, model_b, metric, lam)
        if not a_vals and not b_vals:
            continue
        row(metric, a_vals, b_vals, a_lbl, b_lbl, scale=scale, fmt=fmt)


# ── TabularToy: CEM vs GC-CEM (best-lam run, no lam sweep) ──────────────────

rt = joblib.load("results/results_tabulartoy_suite.dict")
ra = joblib.load("results/results_tabulartoy_acem_suite.dict")
rc = joblib.load("results/results_tabulartoy_cbm_suite.dict")

section("TABULARTOY  CEM vs GC-CEM  (canonical run, no lam level)")
METRICS_CEM = ["task_acc", "c_acc", "cvl", "icvl", "ctl_mean", "icl_mean",
               "probe_acc", "interv"]
compare(rt, "cem", "crcem", METRICS_CEM, lam=None, a_label="CEM", b_label="GC-CEM")

section("TABULARTOY  CEM vs GC-CEM  (lam sweep — acem suite)")
for lam in ["lam_c0.1", "lam_c0.5", "lam_c1"]:
    print(f"\n  [{lam}]")
    compare(ra, "cem", "acem",
            ["task_acc", "c_acc", "cvl", "icvl", "ctl_mean", "icl_mean",
             "probe_acc", "emb_probe_delta", "task_leak", "interv"],
            lam=lam, a_label="CEM", b_label="GC-CEM")

section("TABULARTOY  CBM models  (cbm suite)")
# soft_cbm vs acbm (GC-CBM) at each lam
print("\n  soft_cbm vs GC-CBM (acbm):")
for lam in ["lam_c0.1", "lam_c0.5", "lam_c1"]:
    print(f"  [{lam}]")
    compare(rc, "soft_cbm", "acbm", ["task_acc", "c_acc", "interv"],
            lam=lam, a_label="soft_cbm", b_label="GC-CBM")
print("\n  Other CBM comparisons (lam_c1):")
for model_b in ["hard_cbm", "seq_cbm", "crcbm"]:
    print(f"  soft_cbm vs {model_b}:")
    compare(rc, "soft_cbm", model_b, ["task_acc", "c_acc", "interv"],
            lam="lam_c1", a_label="soft_cbm", b_label=model_b)


# ── dSprites: CEM vs GC-CEM ──────────────────────────────────────────────────

rd  = joblib.load("results/results_dsprites_suite.dict")
rdc = joblib.load("results/results_dsprites_cbm_suite.dict")

section("DSPRITES  CEM vs GC-CEM  (cem suite)")
for lam in ["lam_c0.1", "lam_c0.5", "lam_c1"]:
    print(f"\n  [{lam}]")
    compare(rd, "cem", "crcem",
            ["task_acc", "c_acc", "cvl", "icvl", "ctl_mean", "icl_mean",
             "probe_acc", "icl_probe", "emb_probe_delta", "task_leak",
             "ds_leak", "interv"],
            lam=lam, a_label="CEM", b_label="GC-CEM")

section("DSPRITES  CBM models  (cbm suite)")
print("\n  soft_cbm vs GC-CBM (gc_cbm):")
for lam in ["lam_c0.1", "lam_c0.5", "lam_c1"]:
    print(f"  [{lam}]")
    compare(rdc, "soft_cbm", "gc_cbm", ["task_acc", "c_acc", "interv"],
            lam=lam, a_label="soft_cbm", b_label="GC-CBM")
print("\n  Other CBM comparisons (lam_c1):")
for model_b in ["hard_cbm", "seq_cbm", "crcbm"]:
    print(f"  soft_cbm vs {model_b}:")
    compare(rdc, "soft_cbm", model_b, ["task_acc", "c_acc", "interv"],
            lam="lam_c1", a_label="soft_cbm", b_label=model_b)


# ── CUB: CEM vs GC-CEM ───────────────────────────────────────────────────────

rcu  = joblib.load("results/results_cub_suite.dict")
rcuc = joblib.load("results/results_cub_cbm_suite.dict")

section("CUB  CEM vs GC-CEM  (lam_c0.1 only)")
compare(rcu, "cem", "crcem",
        ["task_acc", "c_acc", "cvl", "icvl", "probe_acc", "icl_probe",
         "emb_probe_delta", "task_leak", "interv"],
        lam="lam_c0.1", a_label="CEM", b_label="GC-CEM")

section("CUB  CBM models  (cbm suite)")
print("\n  soft_cbm vs GC-CBM (gc_cbm):")
for lam in ["lam_c0.1", "lam_c0.5", "lam_c1"]:
    print(f"  [{lam}]")
    compare(rcuc, "soft_cbm", "gc_cbm", ["task_acc", "c_acc", "interv"],
            lam=lam, a_label="soft_cbm", b_label="GC-CBM")
print("\n  Other CBM comparisons (lam_c1):")
for model_b in ["hard_cbm", "seq_cbm"]:
    print(f"  soft_cbm vs {model_b}:")
    compare(rcuc, "soft_cbm", model_b, ["task_acc", "c_acc", "interv"],
            lam="lam_c1", a_label="soft_cbm", b_label=model_b)

print()
print(SEP)
print("Key: *** p<0.001  ** p<0.01  * p<0.05  ns not significant")
print(SEP)
