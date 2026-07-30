# GC-CM: Gradient-Cancelled Concept Models

Concept Bottleneck Models (CBMs) and Concept Embedding Models (CEMs) improve interpretability by routing predictions through a layer of human-defined concepts, enabling interventions and explanations at the concept level. In practice, however, these models suffer from **information leakage**: task-relevant information bypasses the intended concept bottleneck, encoded in the concept representations in ways that are not captured by the concept labels. This undermines interpretability — interventions on concepts have less effect than expected, and the model's behaviour cannot be faithfully explained through concepts alone.

This repository introduces **GC-CM** (Gradient-Cancelled Concept Models), a family of concept-based models designed to suppress leakage at its source. We provide two instantiations:

- **GC-CBM** — a gradient-cancelled variant of the standard Concept Bottleneck Model
- **GC-CEM** — a gradient-cancelled variant of the Concept Embedding Model

The core mechanism is a **Gradient Reversal Layer (GRL)** placed between the concept representations and a shared critic network. During joint training of the concept and task heads, the task loss gradient propagates back into the concept encoder and directly drives leakage — encoding task information in concept activations beyond what the concept labels require. The GRL cancels this task-driven gradient, acting as an adversarial regulariser that prevents the concept encoder from learning task-predictive features that are not concept-aligned.

Code is adapted from the [cem](https://github.com/mateoespinosa/cem) package (see `cem_legacy.md` for adapted files).

---

## Installation

```bash
git clone https://github.com/ahabisaax/GC-CM
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

| Model | `architecture` string | File |
|---|---|---|
| Joint CBM | `ConceptBottleneckModel` | `models/cbm.py` |
| Hard CBM | `IndependentConceptBottleneckModel` | `models/cbm.py` |
| Sequential CBM | `SequentialConceptBottleneckModel` | `models/cbm.py` |
| CEM | `ConceptEmbeddingModel` | `models/cem.py` |
| **GC-CBM** | `GCConceptBottleneckModel` | `models/gc_cbm.py` |
| **GC-CEM** | `GCConceptEmbeddingModel` | `models/gc_cem.py` |

GC-CBM and GC-CEM require `shared_critic: True`. Key hyperparameters: `max_adversarial_lambda` (critic weight), `adversarial_delay` (warm-up epochs before the GRL activates), `n_critic_steps`, `adv_learning_rate`.

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

### CTL and ICL (KSG mutual information, works for CBMs and CEMs)

```bash
python experiments/evaluate_models/evaluate_models_tabulartoy.py
python experiments/evaluate_models/evaluate_models_dsprites.py
python experiments/evaluate_models/evaluate_models_shapes3d.py
```

**CTL** (Concepts-Task Leakage) measures excess mutual information between concept representations and the task label. **ICL** (Interconcept Leakage) measures pairwise mutual information between concept representations. Both use the KSG estimator implemented in `xai_concept_leakage/metrics/mutual_information.py`.

### CVL and ICVL (ridge regression on CEM embeddings)

```bash
python experiments/evaluate_models/compute_tabulartoy_cvl_icvl.py
python experiments/evaluate_models/compute_dsprites_cvl_onehot.py   # one-hot correction for discrete concepts
python experiments/evaluate_models/compute_cub_cvl_icvl.py
```

**CVL** (Concept-Vector Leakage) and **ICVL** (Interconcept-Vector Leakage) are embedding-space analogues of CTL and ICL for CEMs, measuring how much task-predictive and inter-concept information is encoded in the concept embedding vectors via ridge regression.

### Reproduce paper tables

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
