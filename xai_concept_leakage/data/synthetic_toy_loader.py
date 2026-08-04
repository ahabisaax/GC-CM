"""
Synthetic toy dataset for leakage experiments.

Generator adapted from Shared_code_and_results/joint_cbm_frustration/sweep.py.

Dataset config keys:
    n_samples       int   total samples (default 10000)
    n_features      int   input dimensionality r (default 20)
    n_concepts      int   number of *observed* concepts k_known (default 5)
    n_concepts_total int  total concepts k (known + hidden; default 7)
    omega           float weight of hidden concepts on task 0→1 (default 0.3)
    alpha           float inter-concept correlation strength -1→1 (default 0.0)
    sigma_x         float input noise (default 0.5)
    sigma_y         float task label noise (default 0.1)
    alpha_strength  float scale of off-diagonal concept covariance (default 1.0)
    val_frac        float validation fraction (default 0.1)
    test_frac       float test fraction (default 0.2)
    batch_size      int   (default 512)
    num_workers     int   (default 0)
    seed            int   data generation seed (default 42)
"""

import numpy as np
import torch
import torch.nn.functional as F

from xai_concept_leakage.train.utils import extract_dims


# ── Generator (adapted from sweep.py) ────────────────────────────────────────

def _sample_B_components(k, k_known, seed, alpha_strength=1.0):
    rng = np.random.default_rng(seed)
    B_known = np.eye(k_known)
    B_temp = rng.normal(size=(k - k_known, k_known)) * alpha_strength
    assignment = list(range(k_known))
    return B_known, B_temp, assignment, alpha_strength


def _build_B(B_known, B_temp, assignment, alpha, alpha_strength=1.0):
    k_known = B_known.shape[0]
    k = k_known + B_temp.shape[0]
    B = np.zeros((k, k))
    B[:k_known, :k_known] = B_known
    for i, row in enumerate(B_temp):
        j = k_known + i
        col = np.zeros(k_known)
        for idx in range(k_known):
            col[idx] = row[idx] * float(alpha) * alpha_strength
        B[j, :k_known] = col
        B[:k_known, j] = col
    np.fill_diagonal(B, 1.0)
    return B


def generate_toy_dataset(n, r, k, k_known, sigma_x, sigma_y, omega,
                         alpha=0.0, seed=0, alpha_strength=1.0):
    """
    Returns X (n,r), C_known (n,k_known) binarised at 0, y (n,) binary.

    omega: 0 = task depends only on known concepts (no leakage via hidden)
           1 = task depends only on hidden concepts (maximum leakage)
    alpha: inter-concept correlation strength
    """
    B_components = _sample_B_components(k, k_known, seed, alpha_strength)
    B_known, B_temp, assignment, alpha_strength_used = B_components
    B = _build_B(B_known, B_temp, assignment, alpha, alpha_strength_used)

    B_sym = (B + B.T) / 2.0 + 1e-8 * np.eye(k)
    rng = np.random.default_rng(seed)
    Lb = np.linalg.cholesky(B_sym)
    C = rng.normal(size=(n, k)) @ Lb.T          # continuous latent concepts
    A = rng.normal(size=(r, k))
    X = C @ A.T + rng.normal(size=(n, r), scale=float(sigma_x))

    w = rng.normal(size=(k,))
    w_star = w.copy()
    w_star[:k_known] *= (1.0 - omega)
    w_star[k_known:] *= omega
    score = C @ w_star + rng.normal(size=(n,), scale=float(sigma_y))
    y = (score > 0).astype(np.int64)

    C_known = (C[:, :k_known] > 0).astype(np.float32)  # binarise observed concepts
    return X.astype(np.float32), C_known, y


# ── DataLoader builder ────────────────────────────────────────────────────────

def _make_dl(X, C, y, batch_size, num_workers, shuffle=False):
    X_t = torch.FloatTensor(X)
    C_t = torch.FloatTensor(C)
    y_t = torch.LongTensor(y)
    ds = list(zip(X_t, y_t, C_t))
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
    )


def synthetic_toy_dataloaders(config, seed=42):
    n         = config.get("n_samples", 10000)
    r         = config.get("n_features", 20)
    k         = config.get("n_concepts_total", 7)
    k_known   = config.get("n_concepts", 5)
    omega     = float(config.get("omega", 0.3))
    alpha     = float(config.get("alpha", 0.0))
    sigma_x   = float(config.get("sigma_x", 0.5))
    sigma_y   = float(config.get("sigma_y", 0.1))
    alpha_str = float(config.get("alpha_strength", 1.0))
    val_frac  = float(config.get("val_frac", 0.1))
    test_frac = float(config.get("test_frac", 0.2))
    bs        = int(config.get("batch_size", 512))
    nw        = int(config.get("num_workers", 0))
    data_seed = int(config.get("seed", seed))

    X, C, y = generate_toy_dataset(
        n, r, k, k_known, sigma_x, sigma_y, omega,
        alpha=alpha, seed=data_seed, alpha_strength=alpha_str,
    )

    rng = np.random.default_rng(data_seed)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    te_idx = idx[:n_test]
    va_idx = idx[n_test:n_test + n_val]
    tr_idx = idx[n_test + n_val:]

    train_dl = _make_dl(X[tr_idx], C[tr_idx], y[tr_idx], bs, nw, shuffle=True)
    val_dl   = _make_dl(X[va_idx], C[va_idx], y[va_idx], bs, nw)
    test_dl  = _make_dl(X[te_idx], C[te_idx], y[te_idx], bs, nw)
    return train_dl, val_dl, test_dl


# ── Framework entry point ─────────────────────────────────────────────────────

def generate_data(config, root_dir=None, seed=42, output_dataset_vars=False):
    train_dl, val_dl, test_dl = synthetic_toy_dataloaders(config, seed=seed)
    _, n_concepts, n_tasks = extract_dims(train_dl)
    imbalance = None
    if not output_dataset_vars:
        return train_dl, val_dl, test_dl, imbalance
    concept_group_map = {i: [i] for i in range(n_concepts)}
    return train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_group_map)
