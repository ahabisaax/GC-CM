# Reproducibility

Generated from the saved `*_experiment_config.joblib` files and the W&B config
archive, i.e. from what actually ran, not from the repo YAMLs, which have
drifted from several completed runs.

Per-run resolved configs for every run are archived in
`experiments/configs/_archive/<project>/<run_name>__<run_id>.yaml`, each with
its W&B run id, state and URL. Regenerate with:

```bash
python experiments/evaluate_models/export_run_configs.py \
    --projects AwA2 "cub CEM randint" TabularToy_90ep CelebA_redone
```

## Datasets

| dataset | concepts | classes | encoder | emb_size | image | batch | folds |
|---|---|---|---|---|---|---|---|
| TabularToy | 3 | 2 | MLP `{7,64,64,3}` | 16 | n/a | 64 | 5 |
| CUB-200 | 112 | 200 | ResNet-18 (ImageNet) | 32 | 299 | 256 | 5 |
| AwA2 | 85 | 50 | ResNet-34 (ImageNet) | 16 | 128 | 512 | 5 |

`emb_size` applies only to CEM and GC-CEM (per-concept positive/negative
embeddings). It is inert for the CBM family.

AwA2 attributes are **class-level**: `c = predicate_matrix[y]`, so `c_true` is a
deterministic function of the label. ICL and RCL on AwA2 therefore measure
something different from the per-image datasets and are not directly
comparable to CUB.

## Training protocol

Adam throughout. 5 folds (random seeds) per configuration, reported as mean
and standard deviation.

| dataset | model | epochs | lr | early stopping | reported model |
|---|---|---|---|---|---|
| TabularToy | all six | 90 | CEM/CBM/Hard/Seq/GC-CBM 0.05, GC-CEM 1e-3 | `val_y_accuracy` max, patience 400 | final epoch |
| CUB | CEM | 300 | 1e-3 | `val_y_accuracy` max, patience 400 | best checkpoint |
| CUB | GC-CEM | 300 | cbm 5e-4, adv 1e-4 | `val_y_accuracy` max, patience 400 | best checkpoint |
| CUB | Joint / Hard / Seq / GC-CBM | 250 / 300 / 200 / 300 | 1e-4 | mixed, see note | best checkpoint |
| AwA2 | CEM | 90 | 1e-3 | `val_y_accuracy` max, patience 200 | final epoch |
| AwA2 | GC-CEM | 90 | cbm 5e-4, adv 1e-4 | `val_y_accuracy` max, patience 200 | final epoch |
| AwA2 | Joint / GC-CBM | 120 | 1e-3 / cbm 5e-4 | `val_y_accuracy` max, patience 200 | final epoch |
| AwA2 | Hard / Seq | 100 | 1e-3 | `val_y_accuracy` max, patience 200 | final epoch |

Patience always exceeds the number of validation checks, so **early stopping
never fires**; the monitor only selects which checkpoint is saved.
`eval_from_last: true` on TabularToy and AwA2 means the reported model is the
final epoch; CUB reports the best checkpoint.

### Known asymmetries

- **CUB CBM budgets span 200-300 epochs** (Seq 200, Joint 250, Hard and GC-CBM
  300), so GC-CBM trains longer than the baselines it is compared against.
- **AwA2 splits 120 (Joint, GC-CBM) against 100 (Hard, Seq).** The pair
  carrying the comparison is matched; Hard and Seq are fixed-lambda baselines.
- CUB CBM early-stopping monitors differ by model (`val_loss` for Joint and
  Hard, `val_y_accuracy` for Seq and GC-CBM). Since the best checkpoint is
  reported, this changes which weights are evaluated.

## Interventions

Identical everywhere: `random` policy, `group_level: True`, `use_prior: False`,
`intervention_freq: 1`, competence 1.

| dataset | groups | curve points |
|---|---|---|
| TabularToy | 3 concepts | 4 |
| CUB | 28 attribute groups | 29 |
| AwA2 | 17 semantic predicate groups | 18 |

AwA2's 17 groups are defined in `xai_concept_leakage/data/awa2_loader.py` and
verified against `predicates.txt`: all 85 predicates, no overlaps, sizes 1-15.
Group sizes are uneven, so the x-axis is groups revealed, not concepts; plots
use *fraction* intervened, which equals the expected fraction of concepts
because groups are drawn uniformly.

## RandInt

`training_intervention_prob: 0.25`, CEM and GC-CEM only; no CBM RandInt runs.
Every RandInt config differs from its baseline in that single parameter,
verified field-by-field against the baseline's saved config.

Requires commit `566261d`. Before it, `_run_cem_step` and `_run_critic_step`
each called `_after_interventions` a second time, drawing an **independent**
substitution mask, so the task and GRL branches saw different tensors and
their gradients no longer cancelled. Only RandInt runs were affected.

## Leakage metrics

- **RTL / RCL** (CEM embedding space): `metrics/leakage.py`, ridge and MLP
  probes, `global_norm=True`. The MLP probe is capped at 2000 training samples
  (`RTL_MLP_MAX_N`); ridge uses the full training set; the test set is never
  subsampled. Not computed for CBMs, which have no embedding space.
- **CTL / ICL** (concept-probability level): KSG estimator in
  `metrics/mutual_information.py`, on `c_prob`, not raw embeddings.

Report `RTL_sum` and `RTL_norm` explicitly: on CUB they disagree in direction
(`RTL_sum` halves for GC-CEM while `RTL_norm` is flat).

## Superseded runs, do not use

- TabularToy, all models at 240-400 epochs. The CEM baselines also had
  `training_intervention_prob: 0.25`, i.e. they were RandInt runs, not
  baselines. Replaced by the 90-epoch set.
- AwA2 Joint CBM lam_c 0.1 at lr 5e-3 / 90 epochs: 74.05% task accuracy from an
  unstable learning rate. Replaced by lr 1e-3 / 120 epochs.
- AwA2 GC-CBM lam_c 0.1 with folds 1-4 at 90 and fold 5 at 120. Replaced by a
  uniform 120-epoch set (`GCCBM_120ep_*`).
- AwA2 GC-CEM 35-epoch smoke run at batch 1024.
- AwA2 GC-CEM + RandInt single fold predating `566261d`.
- AwA2 Sequential runs predating `8413638`: each fold produced two W&B runs
  with metrics split between them and no intervention curves. Replaced by
  `SeqCBM_rerun_*`.

## Environment

torch 2.3.1+cu121, pytorch-lightning 1.9.5, pinned in `requirements.txt`.
PL 1.x is required: the GC models use the `optimizer_idx` multi-optimizer API
removed in PL 2.0. This also rules out Blackwell GPUs (RTX 5090, `sm_120`),
which torch 2.3.1 cannot target.

Hardware: NVIDIA A40 and RTX A6000 (48GB) on RunPod; A100 (40GB) on the UCL
HPC cluster.

## Fixes that affect results

| commit | effect |
|---|---|
| `566261d` | RandInt: one substitution mask shared by the task and GRL branches |
| `9243d5d` | Sequential/independent CBMs log intervention curves to W&B |
| `8413638` | Sequential logs one W&B run per fold instead of two with split metrics |
| `1199b6d` | Sequential concept-precompute batched instead of `batch_size=1` |
