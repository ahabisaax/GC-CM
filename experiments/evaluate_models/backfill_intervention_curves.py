"""Attach intervention curves to existing W&B runs.

Sequential and independent CBMs computed their intervention curves and saved
them to the split joblibs, but never logged them to W&B: run_experiments
attaches the curves by resuming the training run by id, and only
train_end_to_end_model recorded that id. This backfills the gap for runs that
already finished. Newer runs record the id themselves.

Runs are matched by W&B run name, which follows "<model_name>_fold_<split+1>".

    # see what would be logged, touching nothing
    python experiments/evaluate_models/backfill_intervention_curves.py \
        --results results/awa2_5fold_cbm --project AwA2

    # actually write
    python ... --results ... --project ... --apply
"""
import argparse, glob, os, re, sys
import joblib


def curves_from(path):
    """Return {policy: [acc, ...]} for one split results joblib."""
    try:
        r = joblib.load(path)
    except Exception as e:
        print(f"  ! unreadable: {path} ({e})")
        return {}
    out = {}
    for k, v in r.items():
        if k.endswith("_ints") and isinstance(v, list) and v:
            out[k.replace("test_acc_y_", "").replace("_ints", "")] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="folder holding <model>/<model>_split_N_results.joblib")
    ap.add_argument("--project", required=True, help="W&B project the runs live in")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--apply", action="store_true", help="write to W&B (default: dry run)")
    args = ap.parse_args()

    import wandb
    api = wandb.Api()
    entity = args.entity or api.default_entity
    runs = {}
    dupes = 0
    try:
        for r in api.runs(f"{entity}/{args.project}"):
            # Names can repeat: the sequential path opens a second, short-lived
            # run to log eval metrics rather than resuming the training run.
            # Keep the longer-running one, which is the real training run.
            prev = runs.get(r.name)
            if prev is None:
                runs[r.name] = r
            else:
                dupes += 1
                if (r.summary.get("_runtime") or 0) > (prev.summary.get("_runtime") or 0):
                    runs[r.name] = r
    except Exception as e:
        print(f"could not list project {entity}/{args.project}: {e}")
        sys.exit(1)
    print(f"{len(runs)} distinct run names in {entity}/{args.project}"
          f"{f' ({dupes} duplicate name(s), kept the longer-running one)' if dupes else ''}")

    pat = re.compile(r"(.+)_split_(\d+)_results\.joblib$")
    todo = []
    for path in sorted(glob.glob(os.path.join(args.results, "*", "*_split_*_results.joblib"))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        model, split = m.group(1), int(m.group(2))
        cur = curves_from(path)
        if not cur:
            continue
        name = f"{model}_fold_{split + 1}"
        run = runs.get(name)
        status = "OK" if run else "NO MATCHING W&B RUN"
        pols = ", ".join(f"{p}({len(v)} pts, {v[0]*100:.1f}->{v[-1]*100:.1f}%)" for p, v in cur.items())
        print(f"  [{status}] {name}: {pols}")
        if run:
            todo.append((run, cur))

    if not args.apply:
        print(f"\ndry run: {len(todo)} run(s) would be updated. Re-run with --apply to write.")
        return

    for run, cur in todo:
        try:
            with wandb.init(id=run.id, project=args.project, entity=entity, resume="must") as w:
                log = {}
                for policy, v in cur.items():
                    t = wandb.Table(columns=["n_interventions", "acc"],
                                    data=[[i, float(a)] for i, a in enumerate(v)])
                    log[f"interv/{policy}/curve"] = wandb.plot.line(
                        t, "n_interventions", "acc", title=f"Intervention curve - {policy}")
                    log[f"interv/{policy}/acc_0"] = float(v[0])
                    log[f"interv/{policy}/acc_final"] = float(v[-1])
                w.log(log)
            print(f"  logged {run.name}")
        except Exception as e:
            print(f"  ! failed {run.name}: {e}")


if __name__ == "__main__":
    main()
