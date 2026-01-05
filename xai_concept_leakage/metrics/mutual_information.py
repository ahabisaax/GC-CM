import numpy as np
import torch

##################################################################################################
### Helper functions:
##################################################################################################


def to_numpy(x, copy=True):
    if isinstance(x, np.ndarray):
        return x.copy() if copy else x
    try:
        arr = np.asarray(x)
        return arr.copy() if copy else arr
    except Exception:
        pass

    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        arr = x.detach().cpu().numpy()
        return arr.copy() if copy else arr

    if hasattr(x, "numpy"):
        arr = x.numpy()
        return arr.copy() if copy else arr

    return np.array(x, copy=copy)


def extract_tril(matrix):
    """
    Extracts the lower triangular part of a square matrix
    and returns it as a 1D array. The diagonal is not included.
    """
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def matrix_from_tril(tril):
    """
    Given a 1D array representing the lower or upper triangular part of a square matrix,
    reconstructs the full square matrix. The diagonal is filled with zeros.
    The input array should be of length n*(n-1)/2, where n is the size of the square matrix.
    """
    n_concepts = round(0.5 * (1 + np.sqrt(1 + 8 * len(tril))))
    M = np.zeros((n_concepts, n_concepts))
    M[np.triu_indices(n_concepts, k=1)] = tril
    M += M.T
    return M


def isinteger(numpy_vec):
    # Check if all elements in a numpy array are integers
    return np.all(np.isclose(np.mod(numpy_vec, 1), 0))


##################################################################################################
### MI estimators:
##################################################################################################

from scipy.special import digamma
from sklearn.neighbors import KDTree, NearestNeighbors


def compute_mi_cc(x, y, n_neighbors):
    """
    This function is a modified version of the one in
    sklearn.feature_selection._mutual_info._compute_mi_cc to allow for the
    computation of the mutual information between two one-dimensional or
    multi-dimensional variables.
    Compute mutual information between two continuous variables.

    Parameters
    ----------
    x, y : ndarray, shape (n_samples,)
        Samples of two continuous random variables, must have an identical
        shape.

    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    mi : float
        Estimated mutual information in nat units. If it turned out to be
        negative it is replaced by 0.

    Notes
    -----
    True mutual information can't be negative. If its estimate by a numerical
    method is negative, it means (providing the method is adequate) that the
    mutual information is close to 0 and replacing it by 0 is a reasonable
    strategy.

    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    """
    n_samples = x.size

    """ In sk-learn:
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    """
    if len(x.shape) == 1:
        x = x.reshape((-1, 1))
    if len(y.shape) == 1:
        y = y.reshape((-1, 1))
    xy = np.hstack((x, y))

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=n_neighbors, n_jobs=-1)

    nn.fit(xy)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    kd = KDTree(x, metric="chebyshev")
    nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
    nx = np.array(nx) - 1.0

    kd = KDTree(y, metric="chebyshev")
    ny = kd.query_radius(y, radius, count_only=True, return_distance=False)
    ny = np.array(ny) - 1.0
    mi = (
        digamma(n_samples)
        + digamma(n_neighbors)
        - np.mean(digamma(nx + 1))
        - np.mean(digamma(ny + 1))
    )

    return max(0, mi)


def compute_mi_cc_torch(x, y, k=3):
    """
    Continuous-Continuous MI (KSG) on GPU.
    Uses Chebyshev (L-inf) distance to match standard KSG.
    """

    N = x.shape[0]
    device = x.device

    # Validation & Reshape
    if x.ndim == 1: x = x.view(-1, 1)
    if y.ndim == 1: y = y.view(-1, 1)


    x_norm = x
    y_norm = y

    # Joint vector
    z = torch.cat([x_norm, y_norm], dim=1)

    # Note: 'inf' norm might be slow on some MPS versions.
    # If using Mac M1/M2 and it crashes, switch to p=2.
    p_norm = float('inf')

    # Joint Distances
    dist_z = torch.cdist(z, z, p=p_norm)

    # Find k-th neighbor distance (k+1 because 0 is self)
    values, _ = torch.topk(dist_z, k + 1, largest=False)
    radius = torch.clamp(values[:, -1], min=1e-6)

    # Marginal Counts
    dist_x = torch.cdist(x_norm, x_norm, p=p_norm)
    dist_y = torch.cdist(y_norm, y_norm, p=p_norm)

    nx = (dist_x < radius.unsqueeze(1)).float().sum(dim=1) - 1
    ny = (dist_y < radius.unsqueeze(1)).float().sum(dim=1) - 1
    # Formula
    psi_N = torch.digamma(torch.tensor(N, dtype=torch.float, device=device))
    psi_k = torch.digamma(torch.tensor(k, dtype=torch.float, device=device))
    psi_nx = torch.mean(torch.digamma(nx + 1))
    psi_ny = torch.mean(torch.digamma(ny + 1))

    return max(0.0, (psi_N + psi_k - psi_nx - psi_ny).item())


def compute_mi_cd(c, d, n_neighbors):
    """
    This function is a modified version of the one in
    sklearn.feature_selection._mutual_info._compute_mi_cd to allow for the
    computation of the mutual information between two multi-dimensional variables.
    Compute mutual information between continuous and discrete variables.

    Parameters
    ----------
    c : ndarray, shape (n_samples,)
        Samples of a continuous random variable.

    d : ndarray, shape (n_samples,)
        Samples of a discrete random variable.

    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    mi : float
        Estimated mutual information in nat units. If it turned out to be
        negative it is replaced by 0.

    Notes
    -----
    True mutual information can't be negative. If its estimate by a numerical
    method is negative, it means (providing the method is adequate) that the
    mutual information is close to 0 and replacing it by 0 is a reasonable
    strategy.

    References
    ----------
    .. [1] B. C. Ross "Mutual Information between Discrete and Continuous
       Data Sets". PLoS ONE 9(2), 2014.
    """
    n_samples = c.shape[0]
    """ In sk-learn:
    c = c.reshape((-1, 1))
    """
    if len(c.shape) == 1:
        c = c.reshape((-1, 1))

    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)
    nn = NearestNeighbors(n_jobs=-1)
    for label in np.unique(d):
        mask = d == label
        count = np.sum(mask)
        if count > 1:
            k = min(n_neighbors, count - 1)
            nn.set_params(n_neighbors=k)
            nn.fit(c[mask])
            r = nn.kneighbors()[0]
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = count

    # Ignore points with unique labels.
    mask = label_counts > 1
    n_samples = np.sum(mask)
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    c = c[mask]
    radius = radius[mask]

    kd = KDTree(c)
    m_all = kd.query_radius(c, radius, count_only=True, return_distance=False)
    m_all = np.array(m_all)

    mi = (
        digamma(n_samples)
        + np.mean(digamma(k_all))
        - np.mean(digamma(label_counts))
        - np.mean(digamma(m_all))
    )

    return max(0, mi)


def compute_mi_cd_torch(c, d, k=3):
    """
    Compute MI between continuous c and discrete d on GPU.
    """
    N = c.shape[0]
    device = c.device

    if c.ndim == 1: c = c.view(-1, 1)
    d = d.long().view(-1)

    # Normalize c
    # c_std = c.std(0)
    # c_std[c_std < 1e-6] = 1.0
    # c_norm = (c - c.mean(0)) / c_std
    c_norm = c

    radius = torch.zeros(N, device=device)
    k_counts = torch.zeros(N, device=device)

    # 1. Valid Classes
    unique_labels, counts = torch.unique(d, return_counts=True)
    valid_labels = unique_labels[counts > 1]

    if len(valid_labels) == 0:
        return 0.0

    valid_mask = (d.unsqueeze(1) == valid_labels.unsqueeze(0)).any(dim=1)

    # 2. Conditional Radii
    p_norm = 2
    #    p_norm = float('inf')

    for label in valid_labels:
        mask = (d == label)
        count = mask.sum().item()
        k_eff = min(k, count - 1)

        c_subset = c_norm[mask]
        dist_sub = torch.cdist(c_subset, c_subset, p=p_norm)
        sorted_dists, _ = torch.sort(dist_sub, dim=1)
        r = sorted_dists[:, k_eff]
        radius[mask] = torch.nextafter(r, torch.zeros_like(r))
        k_counts[mask] = float(k_eff)

    # 3. Filter
    c_valid = c_norm[valid_mask]
    r_valid = radius[valid_mask]
    k_valid = k_counts[valid_mask]
    d_valid = d[valid_mask]

    N_valid = c_valid.shape[0]
    if N_valid == 0: return 0.0

    # 4. Global Counts
    dist_full = torch.cdist(c_valid, c_norm, p=p_norm)
    nx = (dist_full < r_valid.unsqueeze(1)).float().sum(dim=1) - 1

    # Safety clamp
    nx = torch.clamp(nx, min=0)

    # 5. Formula
    psi_N = torch.digamma(torch.tensor(N_valid, dtype=torch.float, device=device))
    term_k = torch.mean(torch.digamma(k_valid))
    term_nx = torch.mean(torch.digamma(nx + 1))

    max_label = int(d.max().item())
    count_lookup = torch.zeros(max_label + 1, device=device)
    count_lookup[unique_labels] = counts.float()

    point_class_counts = count_lookup[d_valid]
    term_class = torch.mean(torch.digamma(point_class_counts))

    mi = psi_N + term_k - term_nx - term_class

    if torch.isnan(mi) or torch.isinf(mi):
        return 0.0

    return max(0.0, mi.item())



def compute_cmi_cd(c_1, c_2, d, n_neighbors):
    """
    Computes the Conditional Mutual Information I(c_1, c_2 | y_discrete)
    for continuous c_1, c_2 and discrete y_discrete.

    This is calculated as the expected value of I(c_1, c_2 | Y=y)
    over all classes y.

    Parameters
    ----------
    c_1, c_2 : ndarray, shape (n_samples, n_features)
        Samples of two continuous random variables.
    d : ndarray, shape (n_samples,)
        Samples of a discrete conditioning variable.
    n_neighbours : int
        Number of nearest neighbors for the KSG estimator.

    Returns
    -------
    cmi : float
        Estimated conditional mutual information in nat units.
    """
    total_samples = len(d)
    unique_samples = np.unique(d)
    total_cmi = 0

    for yval in unique_samples:
        indices = np.where(d==yval)[0]
        p_y = len(indices)/total_samples

        if p_y == 0:
            continue

        c_i_subset = c_1[indices]
        c_j_subset = c_2[indices]

        if len(c_i_subset) <= n_neighbors:
            continue
        if c_1.is_cuda:
            mi_subset = compute_mi_cc_torch(c_i_subset, c_j_subset, n_neighbors)
        else:
            mi_subset = compute_mi_cc(c_i_subset, c_j_subset, n_neighbors)

        total_cmi += p_y * mi_subset
    return total_cmi



##################################################################################################
### Wrapper functions for MI estimators:
##################################################################################################

from sklearn.metrics.cluster import mutual_info_score


def estimate_MI_interconcept_gpu(
    c, n_concepts=None, flatten=True, n_neighbors=3, normalise=True
):
    """
    Computes the interconcept mutual information matrix for a set of concept representations.
    Parameters:
    - c :  numpy array or torch tensor of shape (n_samples, n_concepts, emb_dim)
        or (n_samples, n_concepts*emb_dim). In the latter case, n_concepts must be specified.
        emb_dim can be 1 (e.g. in CBM) or >1 (e.g. in CEMs).
    - n_concepts : int.
    - flatten : bool.
        If True, it flattens the lower-triangular part of the output to a 1D array of length
        n_concepts*(n_concepts-1)/2.
    - n_neighbors : int.
        Number of nearest neighbors to use for the MI estimation.
        This is only used if the input is continuous.
    - normalise : bool.
        If True, normalises the MI matrix by dividing by the sqrt of the concept entropies.
    """

    n_samples = c.shape[0]
    if n_concepts is None:
        n_concepts = c.shape[1]

    torch.reshape(c, (n_samples, n_concepts))
    if torch.is_floating_point(c):
        is_integer =  torch.allclose(c, c.floor())
    else:
        is_integer = True


    if is_integer:
        def compute_mi(x, y):
            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()

        # Also sklearn expects 1D arrays for labels, ensure shape is correct
            if x.ndim > 1: x = x.reshape(-1)
            if y.ndim > 1: y = y.reshape(-1)
            return mutual_info_score(x,y)

    else:
        def compute_mi(x, y):
            # We add small noise to have the knn algorithm not fail as suggested in Kraskov et. al.
            noise_x = 1e-4 * torch.mean(x) * torch.randn(*x.shape, device=x.device)
            noise_y = 1e-4 * torch.mean(y) * torch.randn(*y.shape, device=y.device)
            return np.float64(
                compute_mi_cc_torch(x + noise_x, y + noise_y, k=n_neighbors)
            ).item()

    I = np.zeros((n_concepts, n_concepts))
    for ii in range(n_concepts):
        for jj in range(ii + 1, n_concepts):
            I[ii, jj] = compute_mi(c[:, ii], c[:, jj])
    if normalise:
        diag_sqrt_MI = np.sqrt(
            [compute_mi(c[:, ii], c[:, ii]) for ii in range(n_concepts)]
        )
        I /= np.tensordot(diag_sqrt_MI, diag_sqrt_MI, axes=0) + 1e-10
    if flatten:
        output = extract_tril(I)
    else:
        output = I + I.T
    return output

def estimate_MI_interconcept(
    c, n_concepts=None, flatten=True, n_neighbors=3, normalise=True
):
    """
    Computes the interconcept mutual information matrix for a set of concept representations.
    Parameters:
    - c :  numpy array or torch tensor of shape (n_samples, n_concepts, emb_dim)
        or (n_samples, n_concepts*emb_dim). In the latter case, n_concepts must be specified.
        emb_dim can be 1 (e.g. in CBM) or >1 (e.g. in CEMs).
    - n_concepts : int.
    - flatten : bool.
        If True, it flattens the lower-triangular part of the output to a 1D array of length
        n_concepts*(n_concepts-1)/2.
    - n_neighbors : int.
        Number of nearest neighbors to use for the MI estimation.
        This is only used if the input is continuous.
    - normalise : bool.
        If True, normalises the MI matrix by dividing by the sqrt of the concept entropies.
    """
    n_samples = c.shape[0]
    if n_concepts is None:
        n_concepts = c.shape[1]
    c = to_numpy(c)
    c = c.reshape(n_samples, n_concepts, -1)

    if isinteger(c):

        def compute_mi(x, y):
            return mutual_info_score(x.squeeze(-1), y.squeeze(-1))

    else:

        def compute_mi(x, y):
            # We add small noise to have the knn algorithm not fail as suggested in Kraskov et. al.
            noise_x = 1e-10 * np.mean(x) * np.random.randn(*x.shape)
            noise_y = 1e-10 * np.mean(y) * np.random.randn(*y.shape)
            return np.float64(
                compute_mi_cc(x + noise_x, y + noise_y, n_neighbors=n_neighbors)
            ).item()

    I = np.zeros((n_concepts, n_concepts))
    for ii in range(n_concepts):
        for jj in range(ii + 1, n_concepts):
            I[ii, jj] = compute_mi(c[:, ii], c[:, jj])
    if normalise:
        diag_sqrt_MI = np.sqrt(
            [compute_mi(c[:, ii], c[:, ii]) for ii in range(n_concepts)]
        )
        I /= np.tensordot(diag_sqrt_MI, diag_sqrt_MI, axes=0) + 1e-10
    if flatten:
        output = extract_tril(I)
    else:
        output = I + I.T
    return output

def normalised_interconcept_leakage(I, c, n_neighbors=3, n_concepts=None, flatten=True):
    n_samples = c.shape[0]
    if n_concepts is None:
        n_concepts = c.shape[1]

    if isinstance(c, torch.Tensor):
        torch.reshape(c, (n_samples, n_concepts))
        if torch.is_floating_point(c):
            is_integer =  torch.allclose(c, c.floor())
        else:
            is_integer = True
    else:
        c = to_numpy(c)
        c = c.reshape(n_samples, n_concepts, -1)
        is_integer = isinteger(c)

    if is_integer:
        def compute_mi(x, y):
            return mutual_info_score(x.squeeze(-1), y.squeeze(-1))

    else:
        if not isinstance(c, torch.Tensor):
            def compute_mi(x, y):
                # We add small noise to have the knn algorithm not fail as suggested in Kraskov et. al.
                noise_x = 1e-10 * np.mean(x) * np.random.randn(*x.shape)
                noise_y = 1e-10 * np.mean(y) * np.random.randn(*y.shape)
                return np.float64(
                    compute_mi_cc(x + noise_x, y + noise_y, n_neighbors=n_neighbors)
                ).item()
        else:
            def compute_mi(x, y):
                # We add small noise to have the knn algorithm not fail as suggested in Kraskov et. al.
                noise_x = 1e-10 * torch.mean(x) * torch.randn(*x.shape, device=x.device)
                noise_y = 1e-10 * torch.mean(y) * torch.randn(*y.shape, device=y.device)
                return np.float64(
                    compute_mi_cc_torch(x + noise_x, y + noise_y, k=n_neighbors)
                ).item()

    diag_sqrt_MI = np.sqrt(
        [compute_mi(c[:, ii], c[:, ii]) for ii in range(n_concepts)]
    )
    norm_matrix = np.tensordot(diag_sqrt_MI, diag_sqrt_MI, axes=0) + 1e-10
    tril_indices = np.tril_indices(n_concepts, k=-1)
    norm_vector = norm_matrix[tril_indices]
    I_normalized = I / norm_vector

    if flatten:
        return I_normalized
    else:
        output_matrix = np.zeros((n_concepts, n_concepts))
        output_matrix[tril_indices] = I_normalized
        return output_matrix + output_matrix.T


def estimate_cmi_interconcept(c, d, n_concepts=None, flatten=True, n_neighbors=3, normalise=False):
    """
     Computes the conditional interconcept mutual information matrix for a set of concept representations.
     Parameters:
     - c :  numpy array or torch tensor of shape (n_samples, n_concepts, emb_dim)
         or (n_samples, n_concepts*emb_dim). In the latter case, n_concepts must be specified.
         emb_dim can be 1 (e.g. in CBM) or >1 (e.g. in CEMs).
     - n_concepts : int.
     - flatten : bool.
         If True, it flattens the lower-triangular part of the output to a 1D array of length
         n_concepts*(n_concepts-1)/2.
     - n_neighbors : int.
         Number of nearest neighbors to use for the MI estimation.
         This is only used if the input is continuous.
     - normalise : bool.
         If True, normalises the MI matrix by dividing by the sqrt of the concept entropies.
     """
    n_samples = c.shape[0]
    if n_concepts is None:
        n_concepts = c.shape[1]

    if isinstance(c, torch.Tensor):
        torch.reshape(c, (n_samples, n_concepts))
        if torch.is_floating_point(c):
            # Check if all values are equal to their floor (i.e., no decimal part)
            is_integer =  torch.allclose(c, c.floor())
        else:
            is_integer = True
    else:
        c = to_numpy(c)
        c = c.reshape(n_samples, n_concepts, -1)
        is_integer = isinteger(c)

    if is_integer:
        def compute_mi(x, y):
            return mutual_info_score(x.squeeze(-1), y.squeeze(-1))

    else:
        if not isinstance(c, torch.Tensor):
            def compute_mi(x, y):
                # We add small noise to have the knn algorithm not fail as suggested in Kraskov et. al.
                noise_x = 1e-10 * np.mean(x) * np.random.randn(*x.shape)
                noise_y = 1e-10 * np.mean(y) * np.random.randn(*y.shape)
                return np.float64(
                    compute_mi_cc(x + noise_x, y + noise_y, n_neighbors=n_neighbors)
                ).item()
        else:
            def compute_mi(x, y):
                # We add small noise to have the knn algorithm not fail as suggested in Kraskov et. al.
                noise_x = 1e-10 * torch.mean(x) * torch.randn(*x.shape, device=x.device)
                noise_y = 1e-10 * torch.mean(y) * torch.randn(*y.shape, device=y.device)
                return np.float64(
                    compute_mi_cc_torch(x + noise_x, y + noise_y, k=n_neighbors)
                ).item()

    I = np.zeros((n_concepts, n_concepts))
    for ii in range(n_concepts):
        for jj in range(ii + 1, n_concepts):
            I[ii, jj] = compute_cmi_cd(c[:, ii], c[:, jj], d, n_neighbors=n_neighbors)
    # shouldn't normalise
    if normalise:
        diag_sqrt_MI = np.sqrt(
            [compute_mi(c[:, ii], c[:, ii]) for ii in range(n_concepts)]
        )
        I /= np.tensordot(diag_sqrt_MI, diag_sqrt_MI, axes=0) + 1e-10
    if flatten:
        output = extract_tril(I)
    else:
        output = I + I.T
    return output


def repeat_estimate_MI_interconcept(
    c,
    repeats=1,
    return_avg=True,
    n_concepts=None,
    flatten=True,
    n_neighbors=3,
    normalise=True,
):
    """
    Wrapper function for estimate_MI_interconcept to repeat the MI estimation.
    Parameters:
    - c :  numpy array or torch tensor of shape (n_samples, n_concepts, emb_dim)
        or (n_samples, n_concepts*emb_dim). In the latter case, n_concepts must be specified.
        emb_dim can be 1 (e.g. in CBM) or >1 (e.g. in CEMs).
    - repeats : int.
        Number of times to repeat the MI estimation.
    - return_avg : bool.
        If True, it returns the average and standard error of the MI estimation.
        If False, it returns a list of MI estimations.
    - n_concepts : int.
    - flatten : bool.
        If True, it flattens the lower-triangular part of the output to a 1D array of length
        n_concepts*(n_concepts-1)/2.
    - n_neighbors : int.
        Number of nearest neighbors to use for the MI estimation.
        This is only used if the input is continuous.
    - normalise : bool.
        If True, normalises the MI matrix by dividing by the sqrt of the concept entropies.
    """
    Is = [
        estimate_MI_interconcept(
            c,
            n_concepts=n_concepts,
            flatten=flatten,
            n_neighbors=n_neighbors,
            normalise=normalise,
        )
        for _ in range(repeats)
    ]
    if repeats == 1:
        return Is[0]
    else:
        if not return_avg:
            return Is
        else:
            avg = np.mean(Is, axis=0)
            se = np.std(Is, axis=0) / np.sqrt(len(Is) - 1)
            return avg, se


def estimate_MI_concepts_task_gpu(c, y, n_concepts=None, n_neighbors=3, normalise=True):
    """
    Computes the concepts-task mutual information matrix for a set of concept representations.
    Parameters:
    - c :  numpy array or torch tensor of shape (n_samples, n_concepts, emb_dim)
        or (n_samples, n_concepts*emb_dim). In the latter case, n_concepts must be specified.
        emb_dim can be 1 (e.g. in CBM) or >1 (e.g. in CEMs).
    - y :  numpy array or torch tensor of shape (n_samples, n_tasks).
        Task labels. Assuming they are categorical.
    - n_concepts : int.
    - n_neighbors : int.
        Number of nearest neighbors to use for the MI estimation.
        This is only used if the input is continuous.
    - normalise : bool.
        If True, normalises the MI matrix by dividing by the task entropies.
    """
    n_samples = c.shape[0]
    if n_concepts is None:
        n_concepts = c.shape[1]

    # We assume y is always integer:
    def norm_mi(y):
        y_np = y.cpu().detach().numpy()
        return mutual_info_score(y_np, y_np)

    torch.reshape(c, (n_samples, n_concepts))
    if torch.is_floating_point(c):
        is_integer =  torch.allclose(c, c.floor())
    else:
        is_integer = True


    if is_integer:
        def compute_mi(c, y):
            c_np = c.detach().cpu().numpy().squeeze()
            y_np = y.detach().cpu().numpy().squeeze()
            return mutual_info_score(c_np, y_np)
    else:
        def compute_mi(c, y):
            # We add small noise to have the knn algorithm not fail as suggested in Kraskov et. al.
            noise = 5e-5 * torch.mean(c) * torch.randn(*c.shape, device=c.device)
            return np.float64(
                compute_mi_cd_torch(c + noise, y, k=n_neighbors)
            ).item()

    I = np.zeros((n_concepts))
    for ii in range(n_concepts):
        I[ii] = compute_mi(c[:, ii], y)
    if normalise:
        IYY = norm_mi(y)
        I /= IYY
    return I


def estimate_MI_concepts_task(c, y, n_concepts=None, n_neighbors=3, normalise=True):
    """
    Computes the concepts-task mutual information matrix for a set of concept representations.
    Parameters:
    - c :  numpy array or torch tensor of shape (n_samples, n_concepts, emb_dim)
        or (n_samples, n_concepts*emb_dim). In the latter case, n_concepts must be specified.
        emb_dim can be 1 (e.g. in CBM) or >1 (e.g. in CEMs).
    - y :  numpy array or torch tensor of shape (n_samples, n_tasks).
        Task labels. Assuming they are categorical.
    - n_concepts : int.
    - n_neighbors : int.
        Number of nearest neighbors to use for the MI estimation.
        This is only used if the input is continuous.
    - normalise : bool.
        If True, normalises the MI matrix by dividing by the task entropies.
    """
    n_samples = c.shape[0]
    if n_concepts is None:
        n_concepts = c.shape[1]
    y = to_numpy(y)
    c = to_numpy(c)
    c = c.reshape(n_samples, n_concepts, -1)

    # We assume y is always integer:
    def norm_mi(y):
        return mutual_info_score(y, y)

    if isinteger(c):

        def compute_mi(c, y):
            return mutual_info_score(c.squeeze(-1), y)

    else:

        def compute_mi(c, y):
            # We add small noise to have the knn algorithm not fail as suggested in Kraskov et. al.
            noise = 1e-10 * np.mean(c) * np.random.randn(*c.shape)
            return np.float64(
                compute_mi_cd(c + noise, y, n_neighbors=n_neighbors)
            ).item()

    I = np.zeros((n_concepts))
    for ii in range(n_concepts):
        I[ii] = compute_mi(c[:, ii], y)
    if normalise:
        IYY = norm_mi(y)
        I /= IYY
    return I


def compute_MI_score_model_training(
        c_pred,
        c_true,
        y_true,
        apply_max=True,
        score_type="interconcept",
        wrt_true=True,
        n_neighbors=3,
        normalise=True,
        n_concepts=None,
        compute_mi_on_gpu=False):
    if score_type == "interconcept":
        if compute_mi_on_gpu:
            pred_mi = estimate_MI_interconcept_gpu(
                c_pred,
                n_concepts=n_concepts,
                flatten=True,
                n_neighbors=n_neighbors,
                normalise=normalise,
            )

        else:
            pred_mi = estimate_MI_interconcept(
                c_pred,
                n_concepts=n_concepts,
                flatten=True,
                n_neighbors=n_neighbors,
                normalise=normalise,
            )

        if wrt_true:
            if compute_mi_on_gpu:
                true_mi = estimate_MI_interconcept_gpu(
                    c_true,
                    n_concepts=n_concepts,
                    flatten=True,
                    n_neighbors=n_neighbors,
                    normalise=normalise,
                )
            else:
                true_mi = estimate_MI_interconcept(
                    c_true,
                    n_concepts=n_concepts,
                    flatten=True,
                    n_neighbors=n_neighbors,
                    normalise=normalise,
                )
            if apply_max:
                out = np.maximum(0, pred_mi - true_mi)
            else:
                out = pred_mi - true_mi
        else:
            out = pred_mi
    # cmi shouldnt be normalised
    elif score_type == "interconcept_cmi":
        pred_mi = estimate_cmi_interconcept(
            c_pred,
            y_true,
            n_concepts=n_concepts,
            flatten=True,
            n_neighbors=n_neighbors,
            normalise=False,
        )
        if wrt_true:
            true_mi = estimate_cmi_interconcept(
                c_true,
                y_true,
                n_concepts=n_concepts,
                flatten=True,
                n_neighbors=n_neighbors,
                normalise=False,
            )
            if apply_max:
                out = np.maximum(0, pred_mi - true_mi)
            else:
                out = pred_mi - true_mi
        else:
            out = pred_mi

    elif score_type == "concepts_task":
        if compute_mi_on_gpu:
            pred_mi = estimate_MI_concepts_task_gpu(
                c_pred,
                y_true,
                n_concepts=n_concepts,
                n_neighbors=n_neighbors,
                normalise=normalise,
            )
        else:
            pred_mi = estimate_MI_concepts_task(
                c_pred,
                y_true,
                n_concepts=n_concepts,
                n_neighbors=n_neighbors,
                normalise=normalise,
            )
        if wrt_true:
            if compute_mi_on_gpu:
                true_mi = estimate_MI_concepts_task_gpu(
                    c_true,
                    y_true,
                    n_concepts=n_concepts,
                    n_neighbors=n_neighbors,
                    normalise=normalise,
                )
            else:
                true_mi = estimate_MI_concepts_task(
                    c_true,
                    y_true,
                    n_concepts=n_concepts,
                    n_neighbors=n_neighbors,
                    normalise=normalise,
                )
            if apply_max:
                out = np.maximum(0, pred_mi - true_mi)
            else:
                out = pred_mi - true_mi
        else:
            out = pred_mi

    else:
        raise ValueError(f"Unknown score_type: {score_type}")

    return out



def repeat_estimate_MI_concepts_task(
    c, y, repeats=1, return_avg=True, n_concepts=None, n_neighbors=3, normalise=True
):
    """
    Wrapper function for estimate_MI_concepts_task to repeat the MI estimation.
    Parameters:
    - c :  numpy array or torch tensor of shape (n_samples, n_concepts, emb_dim)
        or (n_samples, n_concepts*emb_dim). In the latter case, n_concepts must be specified.
        emb_dim can be 1 (e.g. in CBM) or >1 (e.g. in CEMs).
    - y :  numpy array or torch tensor of shape (n_samples, n_tasks).
        Task labels. Assuming they are categorical.
    - repeats : int.
        Number of times to repeat the MI estimation.
    - return_avg : bool.
        If True, it returns the average and standard error of the MI estimation.
        If False, it returns a list of MI estimations.
    - n_concepts : int.
    - n_neighbors : int.
        Number of nearest neighbors to use for the MI estimation.
        This is only used if the input is continuous.
    - normalise : bool.
        If True, normalises the MI matrix by dividing by the task entropies.
    """
    Is = [
        estimate_MI_concepts_task(
            c, y, n_concepts=n_concepts, n_neighbors=n_neighbors, normalise=normalise
        )
        for _ in range(repeats)
    ]
    if repeats == 1:
        return Is[0]
    else:
        if not return_avg:
            return Is
        else:
            avg = np.mean(Is, axis=0)
            se = np.std(Is, axis=0) / np.sqrt(len(Is) - 1)
            return avg, se


def estimate_MI_two_concept_repr(
    c_1, c_2, n_concepts=None, flatten=False, n_neighbors=3, normalise=True
):
    """
    Computes the mutual information matrix between two sets of concept representations of arbitrary
    dimensions, measuring how predictive each representation in c_1 is of each representation in c_2.
    We assume that the two sets of representations have the same number of samples and concepts,
    and that they are continuous.
    Parameters:
    - c_1:  numpy array or torch tensor of shape (n_samples, n_concepts, emb_dim_1)
        or (n_samples, n_concepts*emb_dim_1). In the latter case, n_concepts must be specified.
    - c_2 :  numpy array or torch tensor of shape (n_samples, n_concepts, emb_dim_2)
        or (n_samples, n_concepts*emb_dim_2). In the latter case, n_concepts must be specified.
    - n_concepts : int.
    - flatten : bool.
        If True, it flattens the lower-triangular part of the output to a 1D array of length
        n_concepts*(n_concepts-1)/2.
    - n_neighbors : int.
        Number of nearest neighbors to use for the MI estimation.
        This is only used if the input is continuous.
    - normalise : bool.
        If True, normalises the MI matrix by dividing by the sqrt of the concept entropies.
    """
    n_samples = c_1.shape[0]
    if n_concepts is None:
        n_concepts = c_1.shape[1]
    c_1 = to_numpy(c_1)
    c_2 = to_numpy(c_2)
    c_1 = c_1.reshape(n_samples, n_concepts, -1)
    c_2 = c_2.reshape(n_samples, n_concepts, -1)

    def compute_mi(x, y):
        # We add small noise to have the knn algorithm not fail as suggested in Kraskov et. al.
        noise_x = 1e-5 * np.mean(x) * np.random.randn(*x.shape)
        noise_y = 1e-5 * np.mean(y) * np.random.randn(*y.shape)
        return np.float64(
            compute_mi_cc(x + noise_x, y + noise_y, n_neighbors=n_neighbors)
        ).item()

    I = np.zeros((n_concepts, n_concepts))
    for ii in range(n_concepts):
        for jj in range(ii + 1, n_concepts):
            I[ii, jj] = compute_mi(c_1[:, ii], c_2[:, jj])
    if normalise:
        diag_sqrt_MI = np.sqrt(
            [compute_mi(c_1[:, ii], c_2[:, ii]) for ii in range(n_concepts)]
        )
        I /= np.tensordot(diag_sqrt_MI, diag_sqrt_MI, axes=0)
    if flatten:
        output = extract_tril(I)
    else:
        output = I + I.T
    return output


##################################################################################################
### Functions to compute correlation matrices and MI from 1- and 2-point marginals:
### (Not used in the paper)
##################################################################################################


def compute_f1(c):
    f1 = np.mean(c, axis=0)
    return f1


def compute_f2(c):
    f1 = compute_f1(c)
    f2 = np.zeros((c.shape[-1], c.shape[-1]))
    for i in range(c.shape[-1]):
        for j in range(c.shape[-1]):
            f2[i, j] = np.dot(c[:, i], c[:, j])
    f2 /= c.shape[0]
    return f2


def compute_C2(c, f1=None, f2=None):
    if f1 is None:
        f1 = compute_f1(c)
    if f2 is None:
        f2 = compute_f2(c)
    C2 = f2 - np.tensordot(f1, f1, axes=0)
    return C2


def compute_MI(c, f1=None, f2=None):
    _EPS = 1e-80
    if f1 is None:
        f1 = compute_f1(c)
    if f2 is None:
        f2 = compute_f2(c)
    MI = f2 * np.log((f2 + _EPS) / (np.tensordot(f1, f1, axes=0) + _EPS))
    return MI


def compute_delta_f1(f1):  # no sqrt(N) at the denominator!
    delta_f1 = (f1 * (1 - f1)) ** 0.5
    return np.absolute(delta_f1)


def compute_delta_f2(f2):  # no sqrt(N) at the denominator!
    delta_f2 = (f2 * (1 - f2)) ** 0.5
    return np.absolute(delta_f2)


def compute_delta_C2(f1, f2, delta_f1=None, delta_f2=None):
    if delta_f1 is None:
        delta_f1 = compute_delta_f1(f1)
    if delta_f2 is None:
        delta_f2 = compute_delta_f2(f2)
    delta_C2 = (
        delta_f2
        + np.tensordot(f1, delta_f1, axes=0)
        + np.tensordot(delta_f1, f1, axes=0)
    )
    return np.absolute(delta_C2)


def compute_delta_MI(f1, f2, delta_f1=None, delta_f2=None):
    _EPS = 1e-30
    if delta_f1 is None:
        delta_f1 = compute_delta_f1(f1)
    if delta_f2 is None:
        delta_f2 = compute_delta_f2(f2)

    delta_MI = delta_f2 * (
        1 + np.log((f2 + _EPS) / (np.tensordot(f1, f1, axes=0) + _EPS))
    )
    delta_MI += (delta_f1 / f1 + np.vstack(delta_f1 / f1)) * f2
    return np.absolute(delta_MI)


def compute_C2_MI_and_deltas(c, compute_MI=False):
    if type(c) == torch.Tensor:
        c = c.numpy()
    f1 = compute_f1(c)
    f2 = compute_f2(c)
    C2 = compute_C2(c, f1, f2)

    delta_f1 = compute_delta_f1(f1)
    delta_f2 = compute_delta_f2(f2)
    delta_C2 = compute_delta_C2(f1, f2, delta_f1, delta_f2)

    if compute_MI:
        MI = compute_MI(c, f1, f2)
        delta_MI = compute_delta_MI(f1, f2, delta_f1, delta_f2)
        return C2, delta_C2 / len(c) ** 0.5, MI, delta_MI / len(c) ** 0.5
    else:
        return C2, delta_C2 / len(c) ** 0.5


def list_strong_couplings(metric, threshold=None, top_percentage=None):
    """
    Given a metric (e.g. MI or C2), returns a list of strong couplings (i,j)
    (i.e. pairs of concepts) that exceed a given threshold.
    If threshold is None, it uses the top_percentage to determine the threshold.
    """
    if threshold is None:
        threshold = np.max(metric) * (1 - top_percentage)
    list_SC = []
    for ii in range(metric.shape[0]):
        for jj in range(ii):
            if metric[ii, jj] > threshold:
                list_SC.append([ii, jj])
    return list_SC


def strong_correlations(
    metric, threshold, i_to_concept, delta_metric=None, threshold_delta=None
):
    """
    Wrapper function for list_strong_couplings to print the strong couplings
    """
    list_couplings = list_strong_couplings(np.absolute(metric), threshold=threshold)
    for i, j in list_couplings:
        if delta_metric is None:
            print(f"{i}, {j}:\t\t {metric[i,j]:.3f}, \t{i_to_concept([i,j])}")
        else:
            if threshold_delta is None:
                print(
                    f"{i}, {j}:\t\t {metric[i,j]:.3f} ± {delta_metric[i,j]:.3f}, \t{i_to_concept([i,j])}"
                )
            elif (
                np.absolute(metric[i, j]) / np.absolute(delta_metric[i, j])
                > threshold_delta
            ):
                print(
                    f"{i}, {j}:\t\t {metric[i,j]:.3f} ± {delta_metric[i,j]:.3f}, \t{i_to_concept([i,j])}"
                )


def t_test_correlations(
    corr_pred, corr_true, n_obs, obs_name_to_index={"C2": 0, "MI": 2}, write=True
):
    p_values = {}
    for observable in obs_name_to_index.keys():
        mean1 = corr_pred[obs_name_to_index[observable]]
        mean2 = corr_true[obs_name_to_index[observable]]
        std1 = corr_pred[obs_name_to_index[observable] + 1] * n_obs**0.5
        std2 = corr_true[obs_name_to_index[observable] + 1] * n_obs**0.5

        _, pvalue = ttest_ind_from_stats(
            mean1=mean1, std1=std1, nobs1=n_obs, mean2=mean2, std2=std2, nobs2=n_obs
        )
        p_values[observable] = pvalue
        if write:
            print(observable + " - pvalues:")
            print(pvalue)
    return p_values


def n_pvalues_below_alpha(p_values, alpha=0.01):
    pvalues = np.copy(p_values)
    pvalues = pvalues - pvalues * np.identity(pvalues.shape[0])
    n_below = (pvalues < alpha).sum() - pvalues.shape[0]
    return n_below // 2


def n_pvalues_below_alpha_dict(p_values, alpha=0.01):
    n_below = {}
    for obs_label, obs in p_values.items():
        n_below[obs_label] = n_pvalues_below_alpha(obs, alpha=alpha)
    return n_below
