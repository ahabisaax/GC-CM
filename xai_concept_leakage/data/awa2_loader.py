"""
Animals with Attributes 2 (AwA2) loader.

AwA2 (Xian et al., 2018) — 37,322 images, 50 animal classes, 85 binary
attributes. Widely used in the concept-bottleneck literature.

Download (no registration required):
    https://cvml.ista.ac.at/AwA2/AwA2-data.zip
Extract to <root_dir>/AwA2/ so that you have:
    <root_dir>/AwA2/Animals_with_Attributes2/
        JPEGImages/<class_name>/<image>.jpg
        classes.txt
        predicates.txt
        predicate-matrix-binary.txt

IMPORTANT — concept granularity
-------------------------------
AwA2 attributes are annotated **per class**, not per image: every zebra
shares one attribute vector. This differs from CUB (per-image annotations)
and CelebA (per-image). Consequences for the leakage metrics here:

  * c_true is a deterministic function of y, so inter-concept mutual
    information is dominated by class structure rather than by anything the
    model learned. ICL/RCL on AwA2 therefore measure something subtly
    different from the per-image datasets and are not directly comparable.
  * Concept accuracy is correspondingly easy — predicting the class implies
    all 85 concepts.

Report AwA2 alongside per-image datasets with that caveat stated, rather
than pooled with them.

Cost note: the MLP RTL/RCL probe performs K^2 fits — 85^2 = 7,225 here,
~4.7x CelebA's 1,521. The RTL_MLP_MAX_N cap (default 2000) keeps this
tractable; see _add_rtl_rcl in train/training.py.
"""

import logging
import os

import numpy as np
import torch
from PIL import Image, ImageFile

# A few AwA2 JPEGs are truncated; allow partial decode rather than
# raising mid-epoch. Genuinely unreadable files are dropped at index time.
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

N_CONCEPTS = 85
N_CLASSES = 50

# Predicate groups in predicates.txt are ordered by semantic family; these
# boundaries follow the grouping used in the original AwA attribute list.
_CONCEPT_GROUP_BOUNDARIES = [
    ("colour_texture", 0, 8),
    ("pattern", 8, 14),
    ("size_shape", 14, 22),
    ("body_parts", 22, 34),
    ("locomotion", 34, 45),
    ("diet", 45, 54),
    ("behaviour", 54, 66),
    ("habitat", 66, 78),
    ("character", 78, 85),
]


def _awa2_root(root_dir):
    """Locate the extracted AwA2 directory under root_dir."""
    for candidate in (
        os.path.join(root_dir, "AwA2", "Animals_with_Attributes2"),
        os.path.join(root_dir, "Animals_with_Attributes2"),
        os.path.join(root_dir, "AwA2"),
    ):
        if os.path.isdir(os.path.join(candidate, "JPEGImages")):
            return candidate
    raise FileNotFoundError(
        f"AwA2 not found under {root_dir}.\n"
        "Download https://cvml.ista.ac.at/AwA2/AwA2-data.zip and extract so that\n"
        f"  {os.path.join(root_dir, 'AwA2', 'Animals_with_Attributes2', 'JPEGImages')}\n"
        "exists."
    )


def _load_metadata(base):
    """Return (class_names, predicate_matrix [50, 85] float32)."""
    with open(os.path.join(base, "classes.txt")) as f:
        class_names = [line.split()[-1].strip() for line in f if line.strip()]
    matrix = np.loadtxt(
        os.path.join(base, "predicate-matrix-binary.txt"), dtype=np.float32
    )
    if matrix.shape != (N_CLASSES, N_CONCEPTS):
        raise ValueError(
            f"predicate-matrix-binary.txt has shape {matrix.shape}, "
            f"expected ({N_CLASSES}, {N_CONCEPTS})"
        )
    return class_names, matrix


def _index_images(base, class_names, verify=True):
    """Return (paths, labels) over all JPEGImages/<class>/*.jpg.

    AwA2 ships a small number of corrupt JPEGs (e.g. collie/collie_10770.jpg),
    which blow up a dataloader worker mid-epoch with PIL.UnidentifiedImageError.
    Rather than fail a multi-hour run, verify headers once at index time and
    drop unreadable files. The scan reads headers only, so it costs seconds.

    The dropped list is cached beside the dataset so later runs skip the scan.
    """
    paths, labels = [], []
    img_root = os.path.join(base, "JPEGImages")
    for label, name in enumerate(class_names):
        class_dir = os.path.join(img_root, name)
        if not os.path.isdir(class_dir):
            logging.warning(f"[awa2_loader] missing class directory: {class_dir}")
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(class_dir, fname))
                labels.append(label)
    if not paths:
        raise RuntimeError(f"No AwA2 images found under {img_root}")

    if verify:
        cache = os.path.join(base, ".awa2_bad_images.txt")
        bad = set()
        if os.path.exists(cache):
            with open(cache) as f:
                bad = {ln.strip() for ln in f if ln.strip()}
            logging.debug(f"[awa2_loader] loaded {len(bad)} known-bad paths from cache")
        else:
            for p in paths:
                try:
                    with Image.open(p) as im:
                        im.verify()          # header/structure only
                except Exception:
                    bad.add(p)
            try:
                with open(cache, "w") as f:
                    f.write("\n".join(sorted(bad)))
            except OSError:
                pass                          # read-only dataset dir is fine
        if bad:
            print(f"[awa2_loader] dropping {len(bad)} unreadable image(s), e.g. "
                  f"{sorted(bad)[0]}")
            keep = [i for i, p in enumerate(paths) if p not in bad]
            paths = [paths[i] for i in keep]
            labels = [labels[i] for i in keep]

    return np.array(paths), np.array(labels, dtype=np.int64)


def _train_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _eval_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class AwA2Dataset(Dataset):
    """Yields (image, label, concepts) — the (x, y, c) contract used here."""

    def __init__(self, paths, labels, predicate_matrix, transform):
        self.paths = paths
        self.labels = labels
        self.predicates = predicate_matrix
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception as e:
            # Belt and braces: the index-time scan should have removed these,
            # but a truncated file can still fail on full decode. Substitute a
            # neighbour of the same class rather than killing the worker.
            logging.warning(f"[awa2_loader] unreadable at load: {self.paths[idx]} ({e})")
            alt = (idx + 1) % len(self.paths)
            img = Image.open(self.paths[alt]).convert("RGB")
            idx = alt
        x = self.transform(img)
        y = int(self.labels[idx])
        c = torch.from_numpy(self.predicates[y].copy())
        return x, y, c


def _build_concept_group_map():
    return {
        i: list(range(start, end))
        for i, (_, start, end) in enumerate(_CONCEPT_GROUP_BOUNDARIES)
    }


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

    Config keys:
        root_dir, batch_size, num_workers, image_size (default 224),
        train_augment, weight_loss, sampling_percent, test_subsampling,
        train_frac / val_frac (class-stratified split; defaults 0.7 / 0.1).
    """
    if seed is not None:
        seed_everything(seed)
    if root_dir is None:
        root_dir = config.get("root_dir", "data/")

    base = _awa2_root(root_dir)
    class_names, predicate_matrix = _load_metadata(base)
    paths, labels = _index_images(base, class_names)

    image_size = config.get("image_size", 224)
    batch_size = config.get("batch_size", 64)
    num_workers = config.get("num_workers", 4)
    do_aug = config.get("train_augment", train_aug)

    # Class-stratified split — AwA2 ships no official train/val/test split.
    rng = np.random.RandomState(42)      # fixed: split must not vary with seed
    train_frac = config.get("train_frac", 0.7)
    val_frac = config.get("val_frac", 0.1)
    tr_idx, va_idx, te_idx = [], [], []
    for label in range(N_CLASSES):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(train_frac * n)
        n_va = int(val_frac * n)
        tr_idx.append(idx[:n_tr])
        va_idx.append(idx[n_tr:n_tr + n_va])
        te_idx.append(idx[n_tr + n_va:])
    tr_idx = np.concatenate(tr_idx)
    va_idx = np.concatenate(va_idx)
    te_idx = np.concatenate(te_idx)

    # Optional subsampling, mirroring the other loaders' knobs.
    sampling_percent = config.get("sampling_percent", 1)
    if sampling_percent != 1:
        keep = int(len(tr_idx) * sampling_percent)
        tr_idx = rng.choice(tr_idx, size=keep, replace=False)
    test_subsampling = config.get("test_subsampling", 1)
    if test_subsampling != 1:
        keep = int(len(te_idx) * test_subsampling)
        te_idx = rng.choice(te_idx, size=keep, replace=False)

    logging.debug(
        f"[awa2_loader] split: {len(tr_idx)} train / {len(va_idx)} val / "
        f"{len(te_idx)} test  ({len(paths)} total)"
    )

    train_ds = AwA2Dataset(paths[tr_idx], labels[tr_idx], predicate_matrix,
                           _train_transform(image_size) if do_aug
                           else _eval_transform(image_size))
    val_ds = AwA2Dataset(paths[va_idx], labels[va_idx], predicate_matrix,
                         _eval_transform(image_size))
    test_ds = AwA2Dataset(paths[te_idx], labels[te_idx], predicate_matrix,
                          _eval_transform(image_size))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)

    if config.get("weight_loss", False):
        train_attrs = predicate_matrix[labels[tr_idx]]
        n_pos = train_attrs.sum(axis=0).clip(min=1)
        imbalance = (len(tr_idx) - n_pos) / n_pos
    else:
        imbalance = None

    if not output_dataset_vars:
        return train_dl, val_dl, test_dl, imbalance

    return train_dl, val_dl, test_dl, imbalance, (
        N_CONCEPTS, N_CLASSES, _build_concept_group_map()
    )
