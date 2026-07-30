"""
Full CEM vs GC-CEM results table.

Sources
-------
- TabularToy : results_tabulartoy_acem_suite.dict  (all 3 λ, all metrics)
- dSprites   : results_dsprites_suite.dict         (λ=0.1 deep metrics)
               + split joblibs                     (all 3 λ for task/conc/CTL/ICL)
- CUB-200    : results_cub_suite.dict              (λ=0.1 error-set only)
               + split joblibs                     (all 3 λ for task/conc/CTL/ICL)

Run from project root:
    python experiments/evaluate_models/generate_full_cem_table.py
"""
import glob, joblib, numpy as np, os, sys
sys.path.insert(0, os.getcwd())

R    = "results/"
lp   = lambda l: "1" if l == 1.0 else str(l)
LAMS = [0.1, 0.5, 1.0]
SPLIT_KEYS = ["test_acc_y", "test_acc_c", "test_ctl_average", "test_icl_average"]


def from_splits(folder, mdir):
    splits = sorted(glob.glob(f"{folder}/{mdir}/*_split_*_results.joblib"))
    if not splits:
        return None
    out = {k: [] for k in SPLIT_KEYS}
    for s in splits:
        d = joblib.load(s)
        for k in SPLIT_KEYS:
            out[k].append(d.get(k, np.nan))
    return {k: np.array(v) for k, v in out.items()}


def ms(arr, pct=False):
    arr = np.asarray(arr, float)
    arr = arr[~np.isnan(arr)]
    if not len(arr):
        return None, None
    m, s = np.mean(arr), np.std(arr)
    if pct:
        return m * 100, s * 100
    return m, s


def fmt(arr, pct=False, dec=1 if True else 3):
    dec = 1 if pct else 3
    m, s = ms(arr, pct)
    if m is None:
        return "--"
    return f"{m:.{dec}f} \\pm {s:.{dec}f}"


# ─── load dicts ───────────────────────────────────────────────────────────────
tt  = joblib.load(R + "results_tabulartoy_acem_suite.dict")
ds  = joblib.load(R + "results_dsprites_suite.dict")
cub = joblib.load(R + "results_cub_suite.dict")

LAM_KEYS = ["lam_c0.1", "lam_c0.5", "lam_c1"]

# ─── helpers to pull suite-dict arrays ────────────────────────────────────────
def tt_arr(label, lk, metric):
    folds = tt[label][lk]
    if metric == "ctl":
        return np.array([np.mean(v["ctl_mean"]) for v in folds.values()])
    if metric == "icl":
        return np.array([v["icl_mean"] for v in folds.values()])
    if metric == "cvl":
        return np.array([v["cvl"]["CVL"] for v in folds.values()])
    if metric == "icvl":
        return np.array([v["icvl"]["ICVL"] for v in folds.values()])
    if metric == "ois":
        return np.array([v["ois_nis_cas"]["ois"] for v in folds.values() if "ois_nis_cas" in v])
    if metric == "nis":
        return np.array([v["ois_nis_cas"]["nis"] for v in folds.values() if "ois_nis_cas" in v])
    if metric == "cas":
        return np.array([v["ois_nis_cas"]["cas"] for v in folds.values() if "ois_nis_cas" in v])
    if metric == "err0":
        return np.array([v["error_set"][0]["task_acc"] for v in folds.values()])
    if metric == "err1":
        return np.array([v["error_set"][1]["task_acc"] for v in folds.values() if 1 in v["error_set"]])
    return np.array([v[metric] for v in folds.values() if metric in v])


def suite_arr(d, label, metric):
    folds = d[label]
    if metric == "ctl":
        return np.array([np.mean(v["ctl_mean"]) for v in folds.values() if "ctl_mean" in v])
    if metric == "icl":
        return np.array([v["icl_mean"] for v in folds.values() if "icl_mean" in v])
    if metric == "cvl":
        return np.array([v["cvl"]["CVL"] for v in folds.values() if "cvl" in v])
    if metric == "icvl":
        return np.array([v["icvl"]["ICVL"] for v in folds.values() if "icvl" in v])
    if metric == "err0":
        return np.array([v["error_set"][0]["task_acc"] for v in folds.values() if "error_set" in v])
    if metric == "err1":
        arr = []
        for v in folds.values():
            if "error_set" in v and 1 in v["error_set"]:
                arr.append(v["error_set"][1]["task_acc"])
        return np.array(arr)
    return np.array([v.get(metric, np.nan) for v in folds.values()])


# ─── print plain-text summary ─────────────────────────────────────────────────
print("=" * 90)
print("CEM vs GC-CEM — full metric summary")
print("=" * 90)

print("\n── TabularToy (all 3 λ) ──")
header = f"{'Model':18s} {'λ':>4s}  {'Task%':>10s}  {'Conc%':>10s}  {'CTL':>8s}  {'ICL':>8s}  "
header += f"{'CVL':>8s}  {'ICVL':>8s}  {'OIS':>6s}  {'NIS':>6s}  {'CAS':>6s}  {'Err@0':>7s}  {'Err@1':>7s}"
print(header)
for lk, lam in zip(LAM_KEYS, LAMS):
    for model, key in [("Joint CEM", "cem"), ("GC-CEM", "acem")]:
        folds = tt[key][lk]
        task = np.array([v["task_acc"] for v in folds.values()])
        cacc = np.array([v["c_acc"] for v in folds.values()])
        row = (f"{model:18s} {lam:>4}  "
               f"{fmt(task,True):>10s}  {fmt(cacc,True):>10s}  "
               f"{fmt(tt_arr(key,lk,'ctl')):>8s}  {fmt(tt_arr(key,lk,'icl')):>8s}  "
               f"{fmt(tt_arr(key,lk,'cvl')):>8s}  {fmt(tt_arr(key,lk,'icvl')):>8s}  "
               f"{fmt(tt_arr(key,lk,'ois')):>6s}  {fmt(tt_arr(key,lk,'nis')):>6s}  "
               f"{fmt(tt_arr(key,lk,'cas')):>6s}  "
               f"{fmt(tt_arr(key,lk,'err0'),True):>7s}  {fmt(tt_arr(key,lk,'err1'),True):>7s}")
        print(row)
    print()

print("\n── dSprites (λ=0.1 for deep metrics; all λ for task/conc/CTL/ICL) ──")
header2 = f"{'Model':18s} {'λ':>4s}  {'Task%':>10s}  {'Conc%':>10s}  {'CTL':>8s}  {'ICL':>8s}  {'CVL*':>8s}  {'ICVL*':>8s}  {'Err@0*':>8s}  {'Err@1*':>8s}"
print(header2)
for lam in LAMS:
    for model, split_folder, split_mdir, suite_key in [
        ("Joint CEM",     R+"dsprites_cem",               "CEM_adam_lr1e-03_bs64_lam_c"+lp(lam),   "cem"),
        ("GC-CEM",        R+"dsprites_acem_shared_critic", "CRCEM_adam_lr5e-05_bs64_lam1_none_lam_c"+lp(lam)+"_shared_critic", "crcem"),
    ]:
        d = from_splits(split_folder, split_mdir)
        task = d["test_acc_y"] if d else np.array([np.nan])
        cacc = d["test_acc_c"] if d else np.array([np.nan])
        ctl  = d["test_ctl_average"] if d else np.array([np.nan])
        icl  = d["test_icl_average"] if d else np.array([np.nan])
        cvl_s  = suite_arr(ds, suite_key, "cvl")  if lam == 0.1 else np.array([])
        icvl_s = suite_arr(ds, suite_key, "icvl") if lam == 0.1 else np.array([])
        err0   = suite_arr(ds, suite_key, "err0") if lam == 0.1 else np.array([])
        err1   = suite_arr(ds, suite_key, "err1") if lam == 0.1 else np.array([])
        row = (f"{model:18s} {lam:>4}  "
               f"{fmt(task,True):>10s}  {fmt(cacc,True):>10s}  "
               f"{fmt(ctl):>8s}  {fmt(icl):>8s}  "
               f"{fmt(cvl_s) if lam==0.1 else '--':>8s}  "
               f"{fmt(icvl_s) if lam==0.1 else '--':>8s}  "
               f"{fmt(err0,True) if lam==0.1 else '--':>8s}  "
               f"{fmt(err1,True) if lam==0.1 else '--':>8s}")
        print(row)
    print()

print("\n── CUB-200 (λ=0.1 for error-set; all λ for task/conc/CTL/ICL; CVL/ICVL≈0) ──")
header3 = f"{'Model':18s} {'λ':>4s}  {'Task%':>10s}  {'Conc%':>10s}  {'CTL':>8s}  {'ICL':>8s}  {'Err@0*':>8s}  {'Err@1*':>8s}"
print(header3)
for lam in LAMS:
    for model, split_mdir, suite_key in [
        ("Joint CEM",  "CEM_adam_lr1e-03_bs256_lam_c"+lp(lam),                          "cem"),
        ("GC-CEM",     "CRCEM_adam_lr5e-04_bs256_lam1_none_lam_c"+lp(lam)+"_shared_critic", "crcem"),
    ]:
        d = from_splits(R+"cub_cem", split_mdir)
        task = d["test_acc_y"] if d else np.array([np.nan])
        cacc = d["test_acc_c"] if d else np.array([np.nan])
        ctl  = d["test_ctl_average"] if d else np.array([np.nan])
        icl  = d["test_icl_average"] if d else np.array([np.nan])
        err0 = suite_arr(cub, suite_key, "err0") if lam == 0.1 else np.array([])
        err1 = suite_arr(cub, suite_key, "err1") if lam == 0.1 else np.array([])
        row = (f"{model:18s} {lam:>4}  "
               f"{fmt(task,True):>10s}  {fmt(cacc,True):>10s}  "
               f"{fmt(ctl):>8s}  {fmt(icl):>8s}  "
               f"{fmt(err0,True) if lam==0.1 else '--':>8s}  "
               f"{fmt(err1,True) if lam==0.1 else '--':>8s}")
        print(row)
    print()

print("* = λ=0.1 only (suite eval was run at a single lambda)")
print("OIS/NIS/CAS = TabularToy only (not run for dSprites/CUB)")
print("CUB CVL/ICVL ≈ 0 (collinearity between y and c at 112 concepts)")
