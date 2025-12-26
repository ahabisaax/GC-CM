import torch
import numpy as np
import os
import sys
import time
import pandas as pd

# Setup Path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from xai_concept_leakage.metrics import mutual_information as mi


# ==============================================================================
# 1. DATA GENERATORS (Edge Cases)
# ==============================================================================

def generate_correlated_data_cc(n_samples, dim, correlation=0.9):
    """
    Continuous-Continuous (ICL):
    X ~ N(0, 1)
    Y ~ corr * X + noise
    Scale Y to be smaller to test normalization robustness.
    Here 'Y' represents a second concept c_j, so it is 1D continuous.
    """
    x = np.random.normal(0, 1, (n_samples, dim))
    noise = np.random.normal(0, 1, (n_samples, dim))
    # Y is correlated but has different scale
    y = (correlation * x + (1 - correlation) * noise) * 0.5

    # Flatten if dim=1 to ensure 1D output for Y as requested
    if dim == 1:
        y = y.flatten()
        x = x.flatten()
    elif dim > 1:
        # For ICL, we usually compare 1D vectors.
        # If dim > 1, it implies vector-valued concepts (CEM).
        # The estimator handles multi-dim X and Y.
        # But for 'y' (as the second variable), let's keep it matching X's dim.
        pass

    return x, y


def generate_collapsed_data_cd(n_samples, dim, n_classes=2):
    """
    Continuous-Discrete (CTL):
    C is 'collapsed' to discrete modes (0.0 or 1.0) with tiny noise.
    This tests the 'Infinite MI' crash on GPU.
    """
    y = np.random.randint(0, n_classes, n_samples)  # 1D Discrete
    # C is effectively discrete (binary) but float type
    c = np.random.randint(0, 2, (n_samples, dim)).astype(float)
    # Add tiny noise (duplicates)
    c += np.random.normal(0, 1e-9, c.shape)
    return c, y


def generate_sparse_class_data(n_samples, dim):
    """
    Continuous-Discrete (CTL):
    One class has very few samples (< k). Tests filtering logic.
    """
    # 99% Class 0, 1% Class 1 (or specifically < 4 samples)
    y = np.zeros(n_samples, dtype=int)  # 1D Discrete
    # Only 2 samples of class 1
    y[-2:] = 1

    c = np.random.normal(0, 1, (n_samples, dim))
    # Make class 1 distinct so it would have high MI if counted
    c[-2:] += 10.0
    return c, y


def generate_cbm_data(n_samples, dim, n_classes=2):
    """
    Generates CBM-style data with:
    - y: 1D Discrete Task Label [N]
    - c_true: Binary Ground Truth Concepts [N, dim]
    - c_pred: Continuous Predicted Concepts [N, dim] (Logits/Probs)
    """
    # 1. Generate Y (1D Discrete)
    y = np.random.randint(0, n_classes, n_samples)

    # 2. Generate C_true (Binary, Multi-dim)
    # Correlate first few concepts with Y
    c_true = np.zeros((n_samples, dim))
    n_relevant = min(dim, 5)

    for k in range(n_relevant):
        # Concept k is active if Y == k (simple dependency)
        c_true[:, k] = (y == (k % n_classes)).astype(float)
        # Add noise (flip labels)
        flip_mask = np.random.rand(n_samples) < 0.1
        c_true[flip_mask, k] = 1 - c_true[flip_mask, k]

    # Remaining concepts random
    for k in range(n_relevant, dim):
        c_true[:, k] = np.random.randint(0, 2, n_samples)

    # 3. Generate C_pred (Continuous, Multi-dim)
    # Start with C_true signal
    c_pred = c_true.copy()
    # Add Gaussian noise to make it continuous (simulating logits/probs)
    c_pred += np.random.normal(0, 0.5, size=c_pred.shape)

    return c_pred, c_true, y


# ==============================================================================
# 2. TEST HARNESS
# ==============================================================================
def run_wrapper_comparison(
    name,
    c_np,
    y_np,
    device,
    n_neighbors=3,
    tolerance=0.05,
    min_corr=0.95,
    top_k=5,
):
    """
    Compare CPU vs GPU wrapper-level MI estimators.
    """

    # CPU
    start = time.time()
    try:
        mi_cpu = mi.estimate_MI_concepts_task(
            c_np, y_np, n_neighbors=n_neighbors, normalise=False
        )
    except Exception as e:
        print(f"{name:<45} | CPU failed: {e}")
        return False
    cpu_time = time.time() - start

    # GPU
    c_t = torch.tensor(c_np, device=device, dtype=torch.float32)
    y_t = torch.tensor(y_np, device=device, dtype=torch.long)

    start = time.time()
    try:
        mi_gpu = mi.estimate_MI_concepts_task_gpu(
            c_t, y_t, n_neighbors=n_neighbors, normalise=False
        )
    except Exception as e:
        print(f"{name:<45} | GPU failed: {e}")
        return False
    gpu_time = time.time() - start

    # Metrics
    diff = np.abs(mi_cpu - mi_gpu)
    max_diff = diff.max()

    # Correlation
    if mi_cpu.std() > 0 and mi_gpu.std() > 0:
        corr = np.corrcoef(mi_cpu, mi_gpu)[0, 1]
    else:
        corr = 1.0

    # Top-k overlap
    top_cpu = set(np.argsort(mi_cpu)[-top_k:])
    top_gpu = set(np.argsort(mi_gpu)[-top_k:])
    topk_overlap = len(top_cpu & top_gpu) / top_k

    status = "✅" if (max_diff < tolerance and corr > min_corr) else "❌"

    print(
        f"{name:<45} | "
        f"maxΔ={max_diff:.4f} | "
        f"corr={corr:.3f} | "
        f"top{top_k}={topk_overlap:.2f} | "
        f"{status}"
    )

    return status == "✅"

def run_comparison(name, cpu_fn, gpu_fn, x_np, y_np, device, tolerance=0.05):
    """
    Generic runner to compare CPU vs GPU outputs.
    """
    # 1. CPU Run
    start = time.time()
    try:
        val_cpu = cpu_fn(x_np, y_np)
    except Exception as e:
        val_cpu = np.nan
        print(f"  [CPU] Failed: {e}")
    cpu_time = time.time() - start

    # 2. GPU Run
    # Handle conversion based on expected input types
    # CTL (Continuous-Discrete): X=Float, Y=Long
    # ICL (Continuous-Continuous): X=Float, Y=Float

    # Heuristic: if Y is integer-like numpy, convert to Long
    if np.issubdtype(y_np.dtype, np.integer):
        y_torch = torch.tensor(y_np, device=device, dtype=torch.long)
    else:
        y_torch = torch.tensor(y_np, device=device, dtype=torch.float32)

    x_torch = torch.tensor(x_np, device=device, dtype=torch.float32)

    start = time.time()
    try:
        # Force synchronization for timing if CUDA
        if device.type == 'cuda': torch.cuda.synchronize()
        val_gpu = gpu_fn(x_torch, y_torch)
        if device.type == 'cuda': torch.cuda.synchronize()

        if isinstance(val_gpu, torch.Tensor):
            val_gpu = val_gpu.cpu().numpy()

    except Exception as e:
        val_gpu = np.nan
        print(f"  [GPU] Failed: {e}")
    gpu_time = time.time() - start

    # 3. Compare
    # Handle array results vs scalar results
    diff = np.abs(val_cpu - val_gpu)
    max_diff = np.max(diff)

    status = "✅" if max_diff < tolerance else "❌"

    # Print Summary
    print(f"{name:<40} | CPU: {np.mean(val_cpu):.4f} | GPU: {np.mean(val_gpu):.4f} | Diff: {max_diff:.4f} | {status}")

    return max_diff < tolerance


def test_suite():
    # Setup Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # MPS doesn't support float64, so comparisons might be slightly looser
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\nRunning Rigorous MI Tests on: {device}")
    print("=" * 100)
    print(f"{'Test Case':<40} | {'CPU Mean':<10} | {'GPU Mean':<10} | {'Max Diff':<10} | Status")
    print("-" * 100)

    # --- SETTINGS ---
    N = 200  # Sample size (keep small for speed, check robustness)
    K = 3  # Neighbors
    dims = [1,2,10,20]




    # --------------------------------------------------------------------------
    # TEST 2: CTL (Continuous-Discrete) Scaling
    # --------------------------------------------------------------------------
    # CPU: compute_mi_cd(c, d, n_neighbors)
    # GPU: compute_mi_cd_gpu(c, d, n_neighbors)

    # Using new CBM data generator
    for d in dims:
        # Use c_pred (continuous) vs y (discrete)
        c_pred, c_true, y = generate_cbm_data(N, dim=d, n_classes=2)

        run_comparison(
            f"CTL (CD) Dim={d} (High Dim C)",
            lambda c, y: mi.compute_mi_cd(c, y, n_neighbors=K),
            lambda c, y: mi.compute_mi_cd_torch(c, y, k=K),
            c_pred, y, device
        )

    # --------------------------------------------------------------------------
    # TEST 3: Edge Case - Collapsed/Binary Data (The "Inf" Bug)
    # --------------------------------------------------------------------------
    c, y = generate_collapsed_data_cd(N, dim=1)
    c += np.random.normal(0, 1e-4, size=c.shape)
    run_comparison(
        "CTL Edge Case: Collapsed/Binary Data",
        lambda c, y: mi.compute_mi_cd(c, y, n_neighbors=K),
        lambda c, y: mi.compute_mi_cd_torch(c, y, k=K),
        c, y, device,
        tolerance=0.1  # Looser tolerance for degenerate data
    )

    # --------------------------------------------------------------------------
    # TEST 4: Edge Case - Sparse Classes (The "Zero Count" Bug)
    # --------------------------------------------------------------------------
    c, y = generate_sparse_class_data(N, dim=3)
    c += np.random.normal(0, 1e-4, size=c.shape)
    run_comparison(
        "CTL Edge Case: Sparse Class (< k)",
        lambda c, y: mi.compute_mi_cd(c, y, n_neighbors=K),
        lambda c, y: mi.compute_mi_cd_torch(c, y, k=K),
        c, y, device
    )

    # --------------------------------------------------------------------------
    # TEST 5: High-Level Wrapper (Vectorized CTL)
    # --------------------------------------------------------------------------
    # This checks if estimate_MI_concepts_task correctly loops/batches over concepts

    # c [N, Concepts], y [N]
    c_pred, c_true, y = generate_cbm_data(N, dim=1, n_classes=2)  # 20 concepts

    # We expect a vector of 20 values
    run_comparison(
        "Wrapper: estimate_MI_concepts_task",
        lambda c, y: mi.estimate_MI_concepts_task(c, y, n_neighbors=K, normalise=False),
        lambda c, y: mi.estimate_MI_concepts_task_gpu(c, y, n_neighbors=K, normalise=False),
        c_pred, y, device
    )

    print("=" * 100)

    for d in [1, 5, 20]:
        x, y_cont = generate_correlated_data_cc(N, dim=d, correlation=0.8)

        # Collapse Y to a scalar task label
        if y_cont.ndim > 1:
            y_scalar = y_cont[:, 0]
        else:
            y_scalar = y_cont

        y = (y_scalar > np.median(y_scalar)).astype(int)

        run_wrapper_comparison(
            f"Wrapper CC-derived → CTL Dim={d}",
            x if x.ndim == 2 else x[:, None],
            y,
            device,
            n_neighbors=K,
        )

        c, y = generate_collapsed_data_cd(N, dim=20)

        # Jitter to restore continuity
        #c += np.random.normal(0, 1e-4, size=c.shape)

        run_wrapper_comparison(
            "Wrapper Collapsed/Binary (jittered)",
            c,
            y,
            device,
            n_neighbors=K,
            tolerance=0.1,
        )

        c, y = generate_sparse_class_data(N, dim=10)

        run_wrapper_comparison(
            "Wrapper Sparse-Class Stress Test",
            c,
            y,
            device,
            n_neighbors=K,
        )


def generate_icl_data(n_samples, dim, correlation=0.8):
    """
    Generates two continuous variables (representing two concepts)
    that are correlated.
    Returns c1 [N, dim], c2 [N, dim]
    """
    # Base signal
    z = np.random.normal(0, 1, (n_samples, dim))

    # c1 is signal + noise
    c1 = z + np.random.normal(0, 0.5, (n_samples, dim))

    # c2 is correlated with c1 (via z)
    # We mix z and independent noise based on correlation strength
    c2 = correlation * z + (1 - correlation) * np.random.normal(0, 0.5, (n_samples, dim))

    return c1, c2


def generate_icl_data(n_samples, dim, correlation=0.8):
    """
    Generates two continuous variables (representing two concepts)
    that are correlated.
    Returns c1 [N, dim], c2 [N, dim]
    """
    z = np.random.normal(0, 1, (n_samples, dim))
    c1 = z + np.random.normal(0, 0.5, (n_samples, dim))
    c2 = correlation * z + (1 - correlation) * np.random.normal(0, 0.5, (n_samples, dim))
    return c1, c2


def generate_mixed_correlation_concepts(n_samples, n_concepts):
    """
    Generates a matrix of concepts where:
    - First half are highly correlated with each other.
    - Second half are independent noise.
    """
    z = np.random.normal(0, 1, (n_samples, 1))
    c_list = []

    # Correlated Group
    for _ in range(n_concepts // 2):
        c_list.append(z + np.random.normal(0, 0.5, (n_samples, 1)))

    # Independent Group
    for _ in range(n_concepts - (n_concepts // 2)):
        c_list.append(np.random.normal(0, 1, (n_samples, 1)))

    return np.hstack(c_list)


def test_icl_suite(device, N=200, K=3):
    print("\n" + "=" * 60)
    print("TEST: Inter-Concept Leakage (ICL) Consistency (Continuous-Continuous)")
    print("=" * 60)

    # 1. Dimensionality Scaling (Low Level)
    print("--- Test 1: Low-Level Scaling (compute_mi_cc) ---")
    dims = [1, 2, 10, 20]  # Removed 50 to save time

    for d in dims:
        c1, c2 = generate_icl_data(N, dim=d, correlation=0.8)
        run_comparison(
            f"ICL (CC) Low-Level Dim={d}",
            lambda x, y: mi.compute_mi_cc(x, y, n_neighbors=K),
            lambda x, y: mi.compute_mi_cc_torch(x, y, k=K),  # Note: using alias
            c1, c2, device
        )

    # 2. Wrapper Test (estimate_MI_interconcept) - Basic 3x3
    print("\n--- Test 2: Wrapper Matrix Structure (3x3) ---")
    c0, c1 = generate_icl_data(N, dim=1, correlation=0.9)
    c2, _ = generate_icl_data(N, dim=1, correlation=0.0)
    c_matrix = np.hstack([c0.reshape(-1, 1), c1.reshape(-1, 1), c2.reshape(-1, 1)])

    start = time.time()
    icl_cpu = mi.estimate_MI_interconcept(c_matrix, n_neighbors=K, normalise=False, flatten=False)
    cpu_time = time.time() - start

    c_torch = torch.tensor(c_matrix, device=device, dtype=torch.float32)
    start = time.time()
    icl_gpu = mi.estimate_MI_interconcept(c_torch, n_neighbors=K, normalise=False, flatten=False)
    if device.type == 'cuda': torch.cuda.synchronize()
    gpu_time = time.time() - start

    if isinstance(icl_gpu, torch.Tensor):
        icl_gpu = icl_gpu.cpu().numpy()

    diff = np.abs(icl_cpu - icl_gpu)
    np.fill_diagonal(diff, 0)
    max_diff = diff.max()
    status = "✅" if max_diff < 0.1 else "❌"

    print(f"Wrapper ICL Matrix [3x3]:")
    print(f"  Max Off-Diagonal Diff: {max_diff:.4f}")
    print(f"  CPU Time: {cpu_time:.4f}s")
    print(f"  GPU Time: {gpu_time:.4f}s")
    print(f"  Status: {status}")

    # 3. Normalization Test
    print("\n--- Test 3: Normalization Logic (normalise=True) ---")
    # Using the same data
    icl_cpu_norm = mi.estimate_MI_interconcept(c_matrix, n_neighbors=K, normalise=True, flatten=False)
    icl_gpu_norm = mi.estimate_MI_interconcept(c_torch, n_neighbors=K, normalise=True, flatten=False)

    if isinstance(icl_gpu_norm, torch.Tensor):
        icl_gpu_norm = icl_gpu_norm.cpu().numpy()

    diff_norm = np.abs(icl_cpu_norm - icl_gpu_norm)
    np.fill_diagonal(diff_norm, 0)  # Diagonals should be 1.0 (or close) if self-normalized correctly?
    # Note: KSG self-MI is an estimator for Entropy, so diagonals aren't strictly 1 unless divided by themselves.
    # The normalization divides by sqrt(H(X)H(Y)). So diag is H(X)/sqrt(H(X)^2) = 1.

    # Check if diagonals are close to 1.0
    diag_mean = np.mean(np.diag(icl_gpu_norm))
    print(f"  GPU Diagonals Mean (Target ~1.0): {diag_mean:.4f}")

    max_diff_norm = diff_norm.max()
    status_norm = "✅" if max_diff_norm < 0.1 else "❌"
    print(f"  Max Diff (Normalized): {max_diff_norm:.4f} | {status_norm}")

    # 4. Flattening Test
    print("\n--- Test 4: Flattening Logic (flatten=True) ---")
    icl_gpu_flat = mi.estimate_MI_interconcept(c_torch, n_neighbors=K, normalise=False, flatten=True)

    expected_len = (3 * 2) // 2  # N*(N-1)/2 = 3
    if icl_gpu_flat.shape[0] == expected_len and icl_gpu_flat.ndim == 1:
        print(f"  Flattened Shape: {icl_gpu_flat.shape} (Expected {expected_len}) | ✅")
    else:
        print(f"  Flattened Shape: {icl_gpu_flat.shape} (Expected {expected_len}) | ❌")

    # 5. Large Scale Matrix Test (20 Concepts)
    print("\n--- Test 5: High-Dim Concepts Matrix (20 Concepts) ---")
    n_concepts_large = 20
    c_mixed = generate_mixed_correlation_concepts(N, n_concepts_large)
    c_mixed_torch = torch.tensor(c_mixed, device=device, dtype=torch.float32)

    start = time.time()
    icl_gpu_large = mi.estimate_MI_interconcept(c_mixed_torch, n_neighbors=K, normalise=False, flatten=False)
    if device.type == 'cuda': torch.cuda.synchronize()
    gpu_time_large = time.time() - start

    if isinstance(icl_gpu_large, torch.Tensor):
        icl_gpu_large = icl_gpu_large.cpu().numpy()

    # Validation: First half should be correlated, second half uncorrelated
    block_corr = icl_gpu_large[:10, :10]
    np.fill_diagonal(block_corr, 0)
    avg_corr = block_corr.mean()

    block_uncorr = icl_gpu_large[10:, 10:]
    np.fill_diagonal(block_uncorr, 0)
    avg_uncorr = block_uncorr.mean()

    print(f"  Matrix Size: {icl_gpu_large.shape}")
    print(f"  Time (20x20): {gpu_time_large:.4f}s")
    print(f"  Avg MI (Correlated Block):   {avg_corr:.4f} (Expected > 0)")
    print(f"  Avg MI (Uncorrelated Block): {avg_uncorr:.4f} (Expected ~ 0)")

    if avg_corr > avg_uncorr + 0.1:
        print("  Structure Check: ✅")
    else:
        print("  Structure Check: ❌ (Correlated block not distinct)")

    # 6. Edge Case: Collapsed Data (Re-verification)
    print("\n--- Test 6: ICL Edge Case: Collapsed Data ---")
    c_bin, _ = generate_collapsed_data_cd(N, dim=2, n_classes=2)
    c1_bin = c_bin[:, 0]
    c2_bin = c_bin[:, 1]

    # Add noise to make it continuous for KSG if needed, or rely on GPU jitter
    c1_bin_noisy = c1_bin + np.random.normal(0, 1e-4, size=c1_bin.shape)
    c2_bin_noisy = c2_bin + np.random.normal(0, 1e-4, size=c2_bin.shape)

    run_comparison(
        "ICL (CC) Collapsed/Binary",
        lambda x, y: mi.compute_mi_cc(x, y, n_neighbors=K),
        lambda x, y: mi.compute_mi_cc_torch(x, y, k=K),
        c1_bin_noisy, c2_bin_noisy, device,
        tolerance=0.1
    )


if __name__ == "__main__":
    device = torch.device('mps')
    test_suite()
    test_icl_suite(device)