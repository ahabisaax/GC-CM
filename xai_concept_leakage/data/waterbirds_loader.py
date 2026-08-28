"""
Waterbirds dataset loader for GC-CEM leakage experiments.

Images come from the Waterbirds distribution-shift benchmark (Sagawa et al. 2020).
Concept annotations are the 112 CUB attributes, matched via shared image filenames.

Expected directory layout
--------------------------
<root_dir>/
  waterbirds/                        ← Waterbirds root (from wilds or manual download)
    metadata.csv
    001.Black_footed_Albatross/
      Black_Footed_Albatross_0001_796111.jpg
      ...
    ...
  CUB200/                            ← existing CUB data
    class_attr_data_10/
      train.pkl
      val.pkl
      test.pkl

metadata.csv columns
---------------------
img_id, y, split, place, img_filename, place_filename
  split  : 0=train, 1=val, 2=test
  y      : 0=landbird, 1=waterbird
  place  : 0=land background, 1=water background
  group  (derived) : y * 2 + place  →  0..3
"""
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_CLASSES   = 2          # landbird / waterbird
N_CONCEPTS  = 112        # CUB 112-attribute set
N_GROUPS    = 4          # (class × background)
IMG_SIZE    = 224        # ResNet-50 standard
SPLIT_MAP   = {"train": 0, "val": 1, "test": 2}


# ---------------------------------------------------------------------------
# Attribute map (CUB pkl → key=class/filename)
# ---------------------------------------------------------------------------

def _build_cub_attr_map(cub_data_dir: str) -> dict:
    """
    Return {relative_img_key: np.ndarray of shape (112,)} mapping from
    last two path components (e.g. '001.Black_footed_Albatross/img.jpg')
    to 112-dim binary attribute vector.
    """
    attr_map = {}
    for split in ("train", "val", "test"):
        pkl = os.path.join(cub_data_dir, "class_attr_data_10", f"{split}.pkl")
        if not os.path.exists(pkl):
            continue
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        for item in data:
            parts = item["img_path"].replace("\\", "/").split("/")
            key = "/".join(parts[-2:])
            attr_map[key] = np.array(item["attribute_label"], dtype=np.float32)
    return attr_map


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WaterbirdsDataset(Dataset):
    """
    Each sample: (image_tensor, y, concept_vector)
    Group labels accessible via self.groups (for worst-group evaluation).
    """

    def __init__(self, metadata_df, waterbirds_root, attr_map, transform=None):
        self.df          = metadata_df.reset_index(drop=True)
        self.root        = waterbirds_root
        self.attr_map    = attr_map
        self.transform   = transform
        # group: 0=landbird/land, 1=landbird/water, 2=waterbird/land, 3=waterbird/water
        self.groups      = (self.df["y"] * 2 + self.df["place"]).values
        self._n_missing  = 0
        self._zero_attr  = np.zeros(N_CONCEPTS, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.root, row["img_filename"])
        img      = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        y = torch.tensor(int(row["y"]), dtype=torch.long)

        # CUB attribute lookup — key is last two path components of img_filename
        parts   = str(row["img_filename"]).replace("\\", "/").split("/")
        key     = "/".join(parts[-2:])
        attrs   = self.attr_map.get(key, None)
        if attrs is None:
            self._n_missing += 1
            attrs = self._zero_attr
        c = torch.tensor(attrs, dtype=torch.float32)

        return img, y, c


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _train_transform():
    return T.Compose([
        T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _eval_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Concept group map (reuse CUB 28-group structure)
# ---------------------------------------------------------------------------

def _build_concept_group_map():
    """28 CUB concept groups, indices 0-111."""
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../data/CUB200"))
        from cub_loader import CONCEPT_GROUP_MAP
        return dict(CONCEPT_GROUP_MAP)
    except Exception:
        # Fallback: one-concept-per-group (no grouping)
        return {str(i): [i] for i in range(N_CONCEPTS)}


# ---------------------------------------------------------------------------
# generate_data — standard interface expected by run_experiments.py
# ---------------------------------------------------------------------------

def generate_data(
    config,
    root_dir=None,
    seed=42,
    output_dataset_vars=False,
    train_aug=True,
    **kwargs,
):
    """
    Returns:
        train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map)
    """
    if seed is not None:
        seed_everything(seed)

    if root_dir is None:
        root_dir = config.get("root_dir", "data/")

    waterbirds_root = os.path.join(root_dir, "waterbirds")
    cub_data_dir    = os.path.join(root_dir, "CUB200")

    metadata_path = os.path.join(waterbirds_root, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Waterbirds metadata.csv not found at {metadata_path}.\n"
            "Download the dataset: https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz\n"
            "and extract to <root_dir>/waterbirds/"
        )

    meta = pd.read_csv(metadata_path)
    attr_map = _build_cub_attr_map(cub_data_dir)

    batch_size  = config.get("batch_size", 64)
    num_workers = config.get("num_workers", 4)
    do_aug      = config.get("train_augment", train_aug)

    splits = {
        "train": meta[meta["split"] == SPLIT_MAP["train"]],
        "val":   meta[meta["split"] == SPLIT_MAP["val"]],
        "test":  meta[meta["split"] == SPLIT_MAP["test"]],
    }

    train_ds = WaterbirdsDataset(splits["train"], waterbirds_root, attr_map,
                                  transform=_train_transform() if do_aug else _eval_transform())
    val_ds   = WaterbirdsDataset(splits["val"],   waterbirds_root, attr_map,
                                  transform=_eval_transform())
    test_ds  = WaterbirdsDataset(splits["test"],  waterbirds_root, attr_map,
                                  transform=_eval_transform())

    n_missing = sum(
        1 for k in (
            "/".join(r["img_filename"].split("/")[-2:])
            for _, r in meta.iterrows()
        ) if k not in attr_map
    )
    if n_missing:
        print(f"  [waterbirds_loader] WARNING: {n_missing} images had no CUB attribute match → zeroed out")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    # Class imbalance (for concept loss weighting if needed)
    if config.get("weight_loss", False):
        all_attrs  = np.stack([attr_map.get(
            "/".join(r["img_filename"].split("/")[-2:]), np.zeros(N_CONCEPTS))
            for _, r in splits["train"].iterrows()])
        n_pos      = all_attrs.sum(axis=0).clip(min=1)
        imbalance  = (len(splits["train"]) - n_pos) / n_pos
    else:
        imbalance = None

    concept_map = _build_concept_group_map()

    if not output_dataset_vars:
        return train_dl, val_dl, test_dl, imbalance

    return train_dl, val_dl, test_dl, imbalance, (N_CONCEPTS, N_CLASSES, concept_map)
