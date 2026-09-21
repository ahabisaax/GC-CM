# TODO

Outstanding experimental work. Datasets in scope for the paper: TabularToy,
CUB-200, AwA2.

## Sequence

1. AwA2 CEM/GC-CEM + RandInt, 5 folds x 3 lambda_c  (running)
2. CUB CEM/GC-CEM + RandInt, 5 folds x 3 lambda_c
   configs: experiments/configs/cub_{cem,gccem}_randint_5fold_lam*.yaml
   300 epochs, emb 32, matched to the CUB baselines. ~30 folds, ~$30-37.
3. The cleanup below.

RandInt is CEM-family only; no CBM RandInt runs are planned.

## Blocking: consistency defects in results already collected

- [ ] **AwA2 GC-CBM lam_c 0.1: re-run folds 1-4 at 120 epochs.**
      Folds 1-4 trained 90 epochs, fold 5 trained 120, so the 5-fold mean
      mixes budgets within one config. Fold 5 was re-run after the configs
      moved to 120. Joint CBM is already uniformly 120 across all lambda_c,
      so this is the only thing stopping Joint and GC-CBM being comparable
      on AwA2. ~4 folds, ~9h, ~$4.40.
      Config: experiments/configs/awa2_gccbm_5fold_lam0_1.yaml (already 120)
      Run with: -p start_split 0 -p trials 4

- [ ] **AwA2 Sequential CBM: fold 5 missing.** Folds 1-4 only; the run hit
      its walltime. Worker is ready: experiments/run_awa2_seqcbm_worker.sh
      (16h, 5 folds, resumes from the first missing fold).

- [ ] **AwA2 Hard and Seq trained 100 epochs against 120 for Joint and
      GC-CBM.** Decide whether to re-run at 120 or report the difference.

- [ ] **TabularToy CBM arm at 90 epochs.** Currently Joint 250, Hard 250,
      Seq 240+100, GC-CBM 300, so GC-CBM gets 20% more training than the
      baselines it is compared against. Evidence they are unnecessary: TT Seq
      CBM reached within 0.2pp of its best validation accuracy by epoch 0-13
      of 239, and the new 90-epoch CEM runs saturate at 99.5-99.8%.
      8 configs (3 Joint + 3 GC-CBM + 1 Hard + 1 Seq), 40 folds, ~$3.
      TT runs at ~2% GPU utilisation, so a CPU pod would be far cheaper.
      Also makes TT consistent with its own CEM arm, now at 90 epochs.

## Deferred by choice

- [ ] **CUB CBM epochs** span 200-300 across models (Joint 250, Hard 300,
      Seq 200+150, GC-CBM 300). Judged acceptable for now; the spread
      favours GC-CBM, so be ready to justify it.

## Known-bad results to discard, not re-use

- AwA2 Joint CBM lam_c 0.1 at lr 5e-3 / 90 epochs: 74.05% task accuracy from
  an unstable learning rate. Superseded by the 1e-3 / 120-epoch runs.
- AwA2 GC-CEM + RandInt, single fold, pre-566261d: the task and GRL branches
  drew independent RandInt masks.
- TabularToy CEM baselines from the old runs had training_intervention_prob
  0.25, so they were RandInt runs, not baselines. Superseded by the 90-epoch
  set.

## Open questions

- [x] **CUB RTL_norm** — RESOLVED (27179a2). Not a missing dict: `RTL_norm`
      divided by the total residual variance, a model-dependent denominator
      that GC-CEM shrinks by construction, so it cancelled the very reduction
      `RTL_sum` showed (6.51 vs 13.31). `leakage.py` now divides by the
      embedding dim d, matching the paper's
      `RTL_k = (1/d) Σ_m max(0, R²_m) σ²_m`. The old quantity is kept as
      `RTL_norm_resid` / `RCL_norm_resid` for comparison. After the fix GC-CEM
      is significantly lower in 22/24 comparisons; the remaining exception is
      genuine — CUB RCL is *higher* for GC-CEM at λ_c 0.5 and 1.0 (p=0.0014,
      3e-6), because the GRL critic predicts the task label and so nothing in
      the objective penalises inter-concept leakage.
- [ ] **Backfill intervention curves** for sequential/independent runs logged
      before 9243d5d:
      `python experiments/evaluate_models/backfill_intervention_curves.py \
          --results results/cub_acbm_shared_critic --project "cub CEM" --apply`
