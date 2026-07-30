# GC-CM: Gradient-Cancelled Concept Models

This repository contains the code for the paper [**Leakage and Interpretability in Concept-Based Models**](https://www.arxiv.org/abs/2504.14094).

We introduce an information-theoretic framework to rigorously characterise and quantify leakage in Concept Bottleneck Models (CBMs) and Concept Embedding Models (CEMs), and propose **GC-CBM** and **GC-CEM** — gradient-cancelled variants that substantially reduce leakage while preserving task performance. We also introduce **CVL** and **ICVL**, embedding-space analogues of our leakage measures for CEMs.

Code is adapted from the [cem](https://github.com/mateoespinosa/cem) package (see `cem_legacy.md` for the list of adapted files).

---

## Installation

```bash
git clone https://github.com/enricoparisini/GC-CM
cd GC-CM
python3 -m pip install .
python3 -c "import xai_concept_leakage"   # verify
```

A Docker image is available:
```bash
docker pull eparisini/xai-concept-leakage
```

---

## Models

All models are instantiated via `construct_model()` in `xai_concept_leakage/models/construction.py` by setting the `architecture` field in a YAML config.

| Paper name | `architecture` string | File |
|---|---|---|
| Joint CBM | `ConceptBottleneckModel` | `models/cbm.py` |
| Hard / Seq CBM | `IndependentConceptBottleneckModel` / `SequentialConceptBottleneckModel` | `models/cbm.py` |
| CEM | `ConceptEmbeddingModel` | `models/cem.py` |
| **GC-CBM** | `GCConceptBottleneckModel` | `models/gc_cbm.py` |
| **GC-CEM** | `GCConceptEmbeddingModel` | `models/gc_cem.py` |

GC-CBM and GC-CEM require `shared_critic: True` in the config to match the paper results. Key adversarial hyperparameters: `max_adversarial_lambda`, `adversarial_delay`, `n_critic_steps`, `adv_learning_rate`.

### Minimal config snippets

**GC-CBM:**
```yaml
- architecture: "GCConceptBottleneckModel"
  use_adversarial: true
  shared_critic: true
  max_adversarial_lambda: 1
  adversarial_delay: 5
  n_critic_steps: 3
  concept_loss_weight: 1.0
```

**GC-CEM:**
```yaml
- architecture: "GCConceptEmbeddingModel"
  emb_size: 16
  use_adversarial: true
  shared_critic: true
  max_adversarial_lambda: 1
  adversarial_delay: 5
  n_critic_steps: 3
  concept_loss_weight: 1.0
```

Full config examples for all datasets are in `experiments/configs/`.

---

## Datasets

| Dataset | How to obtain |
|---|---|
| **TabularToy** | `python data/generate_tabulartoy_dataset.py 0.25 10000` |
| **dSprites** | `cd data && bash download_datasets.sh && cd -` then `python data/generate_dsprites_datasets.py` |
| **3D Shapes** | same download script, then `python data/generate_shapes3d_datasets.py` |
| **CelebA** | manual download required |
| **CUB-200** | manual download required |

The `TabularToy_generation.ipynb` notebook walks through the synthetic data construction.

---

## Training

```bash
# Train all models defined in a config
python experiments/run_experiments.py -c experiments/configs/tabulartoy.yaml

# Override specific parameters
python experiments/run_experiments.py -c experiments/configs/tabulartoy.yaml -p max_epochs=50 batch_size=32

# Filter to a specific model by name
python experiments/run_experiments.py -c experiments/configs/tabulartoy.yaml --filter_in "GCConceptEmbeddingModel"

# Enable W&B logging
python experiments/run_experiments.py -c experiments/configs/tabulartoy.yaml --project_name "my-project"
```

Basic evaluation (task accuracy, concept accuracy, random interventions) runs automatically at the end of each training run.

---

## Leakage Evaluation

### CTL and ICL (KSG-based, works for CBMs and CEMs)

```bash
python experiments/evaluate_models/evaluate_models_tabulartoy.py
python experiments/evaluate_models/evaluate_models_dsprites.py
python experiments/evaluate_models/evaluate_models_shapes3d.py
```

These compute **CTL** (Concepts-Task Leakage) and **ICL** (Interconcept Leakage) for all trained models and save results to `results/`. The underlying KSG mutual information estimator is in `xai_concept_leakage/metrics/mutual_information.py`.

### CVL and ICVL (ridge regression, CEM embedding space only)

```bash
python experiments/evaluate_models/compute_tabulartoy_cvl_icvl.py
python experiments/evaluate_models/compute_dsprites_cvl_onehot.py   # use one-hot for dSprites
python experiments/evaluate_models/compute_cub_cvl_icvl.py
```

**CVL** (Concept-Vector Leakage) and **ICVL** (Interconcept-Vector Leakage) measure how much task-predictive and inter-concept information is encoded in the CEM embedding space beyond the bottleneck, via ridge regression on the concept embedding vectors.

### Reproduce all paper tables

```bash
python experiments/evaluate_models/paper_verify.py      # prints all table values
python experiments/evaluate_models/make_latex_tables.py # outputs LaTeX
```

---

## Results and visualisation

Trained model results are saved to the `results_dir` specified in each config. Example analysis and plots are in:

- `Analyse_results_TabularToy.ipynb` — leakage scores and intervention curves for TabularToy
- `experiments/evaluate_models/plot_*.py` — scripts that reproduce the paper figures
- `experiments/evaluate_models/generate_*.py` / `gen_*.py` — table generation scripts
