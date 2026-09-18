"""Export the resolved config of every W&B run to YAML, for reproducibility.

The per-run *_experiment_config.joblib files live on the pod volumes, which
are not reliably reachable: a stopped RunPod pod is pinned to its original
host and refuses to start when that host's GPUs are let to someone else.
W&B holds the same config for every run and is always reachable, so this is
the durable record.

    python experiments/evaluate_models/export_run_configs.py \
        --projects AwA2 "cub CEM randint" CelebA_redone

Writes experiments/configs/_archive/<project>/<run_name>.yaml, each carrying
the run id, state, created-at and the full config as logged.
"""
import argparse, os, re, sys
import yaml


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects", nargs="+", required=True)
    ap.add_argument("--entity", default=None)
    ap.add_argument("--out", default="experiments/configs/_archive")
    args = ap.parse_args()

    import wandb
    api = wandb.Api()
    entity = args.entity or api.default_entity
    total = 0
    for project in args.projects:
        try:
            runs = list(api.runs(f"{entity}/{project}"))
        except Exception as e:
            print(f"! {project}: {e}")
            continue
        d = os.path.join(args.out, safe(project))
        os.makedirs(d, exist_ok=True)
        n = 0
        for r in runs:
            cfg = dict(r.config or {})
            if not cfg:
                continue
            doc = {
                "_wandb": {
                    "project": project, "entity": entity, "run_id": r.id,
                    "run_name": r.name, "state": r.state,
                    "created_at": str(r.created_at),
                    "url": r.url,
                },
                "config": cfg,
            }
            # several runs can share a name; disambiguate with the id
            path = os.path.join(d, f"{safe(r.name)}__{r.id}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
            n += 1
        print(f"{project}: {n} config(s) -> {d}")
        total += n
    print(f"\n{total} run config(s) exported")


if __name__ == "__main__":
    main()
