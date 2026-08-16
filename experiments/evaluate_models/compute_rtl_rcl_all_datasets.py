"""
Compute RTL and RCL (Ridge and MLP variants) for all datasets and models.

Loads pre-cached embeddings from:
    results/embeddings_tabulartoy_all.joblib
    results/embeddings_dsprites_all.joblib
    results/embeddings_cub_all.joblib

Each cache entry: (c_mix_tr, c_true_tr, y_tr, c_mix_te, c_true_te, y_te)

Saves results to:
    results/results_rtl_rcl_all_datasets.dict

Structure:
    {
      "<dataset>": {
        "<MODEL>_<lam_c>_<fold>": {
          "ridge": {RTL_sum, RTL_norm, RCL_sum, RCL_norm},
          "mlp":   {RTL_sum, RTL_norm, RCL_sum, RCL_norm},
        }, ...
      }, ...
    }

Run from project root:
    python experiments/evaluate_models/compute_rtl_rcl_all_datasets.py
    python experiments/evaluate_models/compute_rtl_rcl_all_datasets.py --datasets tabulartoy dsprites
    python experiments/evaluate_models/compute_rtl_rcl_all_datasets.py --rerun
"""
import argparse, os, sys, time, warnings
warnings.filterwarnings("ignore")

# The probes here are many tiny matmuls. OpenBLAS (DYNAMIC_ARCH, MAX_THREADS=128)
# spawns its full thread pool for each one, so a 640-kFLOP product costs ~1.2s of
# pure scheduling overhead. Capping to one thread is ~640x faster on these shapes.
# Must be set before numpy is imported. Override with RTL_RCL_THREADS if needed
# (note the HPC worker exports OMP/MKL_NUM_THREADS=$NSLOTS, which we deliberately
# override here — more BLAS threads makes this workload slower, not faster).
_nthr = os.environ.get("RTL_RCL_THREADS", "1")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = _nthr

master = os.getcwd().replace("/experiments/evaluate_models", "") + "/"
sys.path.insert(0, master)

import joblib
import numpy as np

from xai_concept_leakage.metrics.leakage import compute_RTL_RCL, compute_RTL_RCL_mlp

CACHE_PATHS = {
    "tabulartoy": master + "results/embeddings_tabulartoy_all.joblib",
    "dsprites":   master + "results/embeddings_dsprites_all.joblib",
    "cub":        master + "results/embeddings_cub_all.joblib",
}
SAVE_PATH = master + "results/results_rtl_rcl_all_datasets.dict"

MLP_HIDDEN = (64,)   # one hidden layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["tabulartoy", "dsprites", "cub"],
                    choices=["tabulartoy", "dsprites", "cub"])
    ap.add_argument("--rerun", action="store_true",
                    help="Recompute even if already cached in results dict")
    ap.add_argument("--skip_mlp", action="store_true",
                    help="Skip MLP variant (useful for CUB K=112 speed test)")
    args = ap.parse_args()

    results = joblib.load(SAVE_PATH) if os.path.exists(SAVE_PATH) else {}

    for dataset in args.datasets:
        cache_path = CACHE_PATHS[dataset]
        if not os.path.exists(cache_path):
            print(f"[{dataset}] SKIP — cache not found: {cache_path}")
            continue

        print(f"\n{'='*64}")
        print(f"  Dataset: {dataset}")
        print(f"{'='*64}")
        emb_cache = joblib.load(cache_path)
        results.setdefault(dataset, {})

        keys = sorted(emb_cache.keys())
        print(f"  {len(keys)} cached entries: {keys[:3]} ...")

        for i, key in enumerate(keys, 1):
            entry = results[dataset].get(key, {})
            need_ridge = "ridge" not in entry or args.rerun
            need_mlp   = ("mlp" not in entry or args.rerun) and not args.skip_mlp

            if not need_ridge and not need_mlp:
                r = entry.get("ridge", {})
                m = entry.get("mlp", {})
                print(f"  [{i:2d}/{len(keys)}] {key}  [cached]"
                      f"  ridge RTL={r.get('RTL_sum','?'):.4f}"
                      f"  mlp RTL={m.get('RTL_sum','?'):.4f}")
                continue

            print(f"\n  [{i:2d}/{len(keys)}] {key}")
            c_mix_tr, c_true_tr, y_tr, c_mix_te, c_true_te, y_te = emb_cache[key]
            K, m = c_mix_tr.shape[1], c_mix_tr.shape[2]
            print(f"    emb_tr={c_mix_tr.shape}  emb_te={c_mix_te.shape}"
                  f"  K={K}  m={m}  L={len(np.unique(y_tr))}", flush=True)

            args_tuple = (c_mix_tr, c_mix_te, c_true_tr, c_true_te, y_tr, y_te)

            if need_ridge:
                t0 = time.time()
                print(f"    Ridge ...", end=" ", flush=True)
                entry["ridge"] = compute_RTL_RCL(*args_tuple)
                r = entry["ridge"]
                print(f"RTL_sum={r['RTL_sum']:.4f}  RTL_norm={r['RTL_norm']:.4f}"
                      f"  RCL_sum={r['RCL_sum']:.4f}  RCL_norm={r['RCL_norm']:.4f}"
                      f"  ({time.time()-t0:.1f}s)", flush=True)

            if need_mlp:
                t0 = time.time()
                print(f"    MLP{MLP_HIDDEN} ...", end=" ", flush=True)
                entry["mlp"] = compute_RTL_RCL_mlp(*args_tuple, hidden=MLP_HIDDEN)
                r = entry["mlp"]
                print(f"RTL_sum={r['RTL_sum']:.4f}  RTL_norm={r['RTL_norm']:.4f}"
                      f"  RCL_sum={r['RCL_sum']:.4f}  RCL_norm={r['RCL_norm']:.4f}"
                      f"  ({time.time()-t0:.1f}s)", flush=True)

            results[dataset][key] = entry
            joblib.dump(results, SAVE_PATH)

    print(f"\n{'='*64}")
    print("Summary")
    print(f"{'='*64}")
    for ds in args.datasets:
        if ds not in results:
            continue
        print(f"\n  {ds}")
        for key, entry in sorted(results[ds].items()):
            parts = []
            for probe in ["ridge", "mlp"]:
                if probe in entry:
                    r = entry[probe]
                    parts.append(
                        f"{probe}: RTL_sum={r['RTL_sum']:.4f}"
                        f" norm={r['RTL_norm']:.4f}"
                        f" RCL_sum={r['RCL_sum']:.4f}"
                        f" norm={r['RCL_norm']:.4f}"
                    )
            print(f"    {key}:  " + "  |  ".join(parts))

    print(f"\nSaved → {SAVE_PATH}")


if __name__ == "__main__":
    main()
