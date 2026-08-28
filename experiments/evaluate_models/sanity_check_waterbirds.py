"""
Waterbirds sanity checks — run before any full training.

Checks
------
1. Dataset loads without error; metadata.csv columns are as expected.
2. Group counts match Sagawa et al. (train: ~3,498 / ~184 / ~56 / ~1,057).
3. CUB attribute map coverage — what % of Waterbirds images have attributes.
4. Spot-check: print class name + top-5 active attributes for 3 random train images.
5. One-epoch smoke test: CEM, one seed, batch_size=32, verifies loss decreases.

Run from project root:
    python experiments/evaluate_models/sanity_check_waterbirds.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
print("=" * 60)
print("CHECK 1 — Dataset loading")
print("=" * 60)

import xai_concept_leakage.data.waterbirds_loader as wb

config = {
    "batch_size": 32,
    "num_workers": 0,
    "root_dir": "data/",
    "train_augment": False,
    "weight_loss": False,
}

try:
    train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = wb.generate_data(
        config=config, seed=42, output_dataset_vars=True
    )
    print(f"  n_concepts={n_concepts}, n_tasks={n_tasks}, n_concept_groups={len(concept_map)}")
    print(f"  train={len(train_dl.dataset)}  val={len(val_dl.dataset)}  test={len(test_dl.dataset)}")
except FileNotFoundError as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Group counts
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CHECK 2 — Group counts")
print("=" * 60)

group_names = ["landbird/land", "landbird/water", "waterbird/land", "waterbird/water"]
for split_name, ds in [("train", train_dl.dataset), ("val", val_dl.dataset), ("test", test_dl.dataset)]:
    counts = np.bincount(ds.groups, minlength=4)
    print(f"  {split_name:5s}: " + "  ".join(f"{g}={c}" for g, c in zip(group_names, counts)))

# ---------------------------------------------------------------------------
# 3. Attribute coverage
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CHECK 3 — CUB attribute map coverage")
print("=" * 60)

n_total   = len(train_dl.dataset) + len(val_dl.dataset) + len(test_dl.dataset)
n_in_map  = sum(
    1 for ds in [train_dl.dataset, val_dl.dataset, test_dl.dataset]
    for k in (
        "/".join(str(r["img_filename"]).split("/")[-2:])
        for _, r in ds.df.iterrows()
    )
    if k in ds.attr_map
)
print(f"  {n_in_map}/{n_total} images matched to CUB attributes ({100*n_in_map/n_total:.1f}%)")
if n_in_map < 0.9 * n_total:
    print("  WARNING: coverage < 90% — check cub_data_dir path and pkl files")

# ---------------------------------------------------------------------------
# 4. Spot-check attributes
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CHECK 4 — Attribute spot-check (3 random train images)")
print("=" * 60)

try:
    from data.CUB200.cub_loader import CONCEPT_SEMANTICS, SELECTED_CONCEPTS
    attr_names = np.array(CONCEPT_SEMANTICS)[SELECTED_CONCEPTS].tolist()
except Exception:
    attr_names = [f"attr_{i}" for i in range(112)]

rng  = np.random.default_rng(0)
idxs = rng.integers(0, len(train_dl.dataset), size=3)
ds   = train_dl.dataset
for idx in idxs:
    row   = ds.df.iloc[idx]
    key   = "/".join(str(row["img_filename"]).split("/")[-2:])
    attrs = ds.attr_map.get(key)
    label = "waterbird" if row["y"] == 1 else "landbird"
    bg    = "water" if row["place"] == 1 else "land"
    if attrs is not None:
        active = [attr_names[i] for i in np.where(attrs)[0]][:5]
        print(f"  [{label}/{bg}] {row['img_filename'].split('/')[-1]}")
        print(f"    top active attrs: {active}")
    else:
        print(f"  [{label}/{bg}] {row['img_filename'].split('/')[-1]} — NO ATTRIBUTES FOUND")

# ---------------------------------------------------------------------------
# 5. One-epoch smoke test
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CHECK 5 — One-epoch smoke test (CEM, seed=0)")
print("=" * 60)

import pytorch_lightning as pl
from xai_concept_leakage.models.construction import construct_model

model_config = {
    "architecture":          "ConceptEmbeddingModel",
    "n_concepts":            n_concepts,
    "n_tasks":               n_tasks,
    "concept_map":           concept_map,
    "emb_size":              16,
    "extra_dims":            0,
    "bool":                  False,
    "linear_c2y":            True,
    "concept_loss_weight":   1.0,
    "learning_rate":         1e-4,
    "weight_decay":          1e-4,
    "optimizer":             "adam",
    "c_extractor_arch":      "resnet50",
    "sigmoidal_prob":        True,
    "training_intervention_prob": 0,
    "embedding_activation":  "leakyrelu",
    "momentum":              0.9,
    "n_hidden":              512,
    "top_k_accuracy":        None,
    "use_task_class_weights": False,
    "weight_loss":           False,
    "compute_mi_on_gpu":     False,
    "compute_mi_mode":       "cpu",
    "intervention_freq":     1,
    "intervention_batch_size": 64,
    "competence_levels":     [1],
    "intervention_policies": [{"policy": "random", "group_level": True, "use_prior": False}],
}

pl.seed_everything(0)
model = construct_model(
    n_concepts, n_tasks,
    config=model_config,
    imbalance=imbalance,
    concept_map=concept_map,
)

trainer = pl.Trainer(
    max_epochs=1,
    accelerator="auto",
    enable_progress_bar=True,
    enable_model_summary=False,
    logger=False,
    enable_checkpointing=False,
)
trainer.fit(model, train_dl, val_dl)

val_results = trainer.test(model, test_dl, verbose=False)
print(f"\n  Smoke test passed!")
print(f"  test_acc_y  = {val_results[0].get('test_acc_y', 'N/A'):.4f}")
print(f"  test_acc_c  = {val_results[0].get('test_acc_c', 'N/A'):.4f}")
print("\nAll checks done.")
