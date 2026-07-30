"""Run paired t-tests on per-fold metrics for TabularToy and dSprites."""
import joblib, numpy as np, os, sys
from scipy import stats

sys.path.insert(0, os.getcwd())


def get_metric(r, model, key, lam=None):
    folds = r[model] if lam is None else r[model].get(lam, {})
    vals = []
    for f, d in folds.items():
        if not isinstance(d, dict):
            continue
        v = d.get(key)
        if v is None:
            continue
        if isinstance(v, dict):
            v = v.get("CVL", v.get("ICVL", v.get("CVL_global", None)))
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals


def ttest_report(label, a_vals, b_vals, a_name, b_name):
    a, b = np.array(a_vals), np.array(b_vals)
    if len(a) < 2 or len(b) < 2:
        print("  {}: insufficient data".format(label))
        return
    t, p = stats.ttest_ind(a, b)
    sig = "  *" if p < 0.05 else ""
    print(
        "  {:12s}:  {} {:.3f}+/-{:.3f}  vs  {} {:.3f}+/-{:.3f}  |  t={:.2f}  p={:.3f}{}".format(
            label,
            a_name, np.mean(a), np.std(a),
            b_name, np.mean(b), np.std(b),
            t, p, sig,
        )
    )


# ---- TabularToy ----
rt = joblib.load("results/results_tabulartoy_suite.dict")
print("=== TabularToy (CEM vs GC-CEM) ===")
for key, label in [
    ("task_acc", "Task acc"),
    ("c_acc", "Concept acc"),
    ("cvl", "CVL"),
    ("icvl", "ICVL"),
]:
    a = get_metric(rt, "cem", key)
    b = get_metric(rt, "crcem", key)
    ttest_report(label, a, b, "CEM", "GC-CEM")

# ---- dSprites ----
rd = joblib.load("results/results_dsprites_suite.dict")
print()
print("=== dSprites (CEM vs GC-CEM, lam_c0.1) ===")
for key, label in [
    ("task_acc", "Task acc"),
    ("c_acc", "Concept acc"),
    ("cvl", "CVL"),
    ("icvl", "ICVL"),
]:
    a = get_metric(rd, "cem", key, lam="lam_c0.1")
    b = get_metric(rd, "crcem", key, lam="lam_c0.1")
    ttest_report(label, a, b, "CEM", "GC-CEM")

# ---- also check how many folds ----
print()
print("Fold counts:")
print("  TT CEM:", len(get_metric(rt, "cem", "task_acc")),
      "  TT GC-CEM:", len(get_metric(rt, "crcem", "task_acc")))
print("  dSprites CEM:", len(get_metric(rd, "cem", "task_acc", lam="lam_c0.1")),
      "  dSprites GC-CEM:", len(get_metric(rd, "crcem", "task_acc", lam="lam_c0.1")))
