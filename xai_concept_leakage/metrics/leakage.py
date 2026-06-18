"""
Regression-based leakage metrics for CEM concept embeddings.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


def compute_CVL(
    c_hat_train,
    c_hat_test,
    c_train,
    c_test,
    y_train,
    y_test,
    alpha=1.0,
    C=1.0,
    max_iter=500,
):
    """
    Concept Vector Leakage (CVL).

    For each concept k:
      1. Ridge(c_k → ĉ_k [N, m]) → residuals R_k
         Removes variance in the embedding linearly explained by k's own label.
      2. Ridge(y → R_k): fit task label as predictor of the residual embedding.
         Compute variance-weighted R² across embedding dimensions.
         CVL_k = max(0, R²)

    If R² is positive, the residuals are structured by y — leakage is present.

    Parameters
    ----------
    c_hat_train : array [N_train, K, m]
    c_hat_test  : array [N_test,  K, m]
    c_train     : array [N_train, K]  — binary ground-truth concept labels
    c_test      : array [N_test,  K]
    y_train     : array [N_train]     — integer task labels
    y_test      : array [N_test]
    alpha       : float  Ridge regularisation for both steps (default 1.0)
    C, max_iter : unused, kept for API compatibility

    Returns
    -------
    dict with keys:
        'CVL'             : float        — mean clipped R²; near 0 = clean
        'CVL_per_concept' : list[float]  — clipped R² per concept (length K)
        'r2_per_concept'  : list[float]  — raw variance-weighted R² per concept
    """
    c_hat_train = np.asarray(c_hat_train)
    c_hat_test  = np.asarray(c_hat_test)
    c_train     = np.asarray(c_train)
    c_test      = np.asarray(c_test)
    y_train     = np.asarray(y_train).ravel()
    y_test      = np.asarray(y_test).ravel()

    K = c_hat_train.shape[1]

    cvl_per_concept = []
    r2_per_concept  = []
    for k in range(K):
        # Step 1: Ridge(c_k → ĉ_k) → residuals orthogonal to c_k
        ridge1 = Ridge(alpha=alpha)
        ridge1.fit(c_train[:, k].reshape(-1, 1), c_hat_train[:, k, :])
        R_train = c_hat_train[:, k, :] - ridge1.predict(c_train[:, k].reshape(-1, 1))
        R_test  = c_hat_test[:, k, :]  - ridge1.predict(c_test[:, k].reshape(-1, 1))

        # Step 2: Ridge(y → R_k); variance-weighted R² across embedding dims
        ridge2 = Ridge(alpha=alpha)
        ridge2.fit(y_train.reshape(-1, 1), R_train)
        R_pred = ridge2.predict(y_test.reshape(-1, 1))
        r2    = float(r2_score(R_test, R_pred, multioutput="variance_weighted"))
        delta = max(0.0, r2)
        print(f"      concept {k:3d}: r2={r2:.4f}  CVL={delta:.4f}")
        r2_per_concept.append(r2)
        cvl_per_concept.append(delta)

    return {
        "CVL":             float(np.mean(cvl_per_concept)),
        "CVL_per_concept": cvl_per_concept,
        "r2_per_concept":  r2_per_concept,
    }


def compute_CVL_global(
    c_hat_train,
    c_hat_test,
    c_train,
    c_test,
    y_train,
    y_test,
    alpha=1.0,
    C=1.0,
    max_iter=500,
):
    """
    Global Concept Variance Leakage (CVL_global).

    A single regression over the full concatenated embedding [N, K*m] rather
    than K separate per-concept regressions.  More stable for large K (e.g. CUB).

    Steps:
      1. Ridge(c_train [N,K] → c_hat.reshape(N,K*m)) → residuals R
         Removes all linearly concept-explained variance from the full embedding.
      2. Ridge(R → y) → R² as CVL_global
      CVL_global = max(0, R²)

    Parameters
    ----------
    c_hat_train : array [N_train, K, m]
    c_hat_test  : array [N_test,  K, m]
    c_train     : array [N_train, K]  — binary ground-truth concept labels
    c_test      : array [N_test,  K]
    y_train     : array [N_train]     — integer task labels
    y_test      : array [N_test]
    alpha       : float  Ridge regularisation for both steps (default 1.0)
    C           : unused, kept for API compatibility
    max_iter    : unused, kept for API compatibility

    Returns
    -------
    dict with keys:
        'CVL_global' : float — clipped R²; near 0 = clean
        'r2'         : float — raw R²
    """
    c_hat_train = np.asarray(c_hat_train)
    c_hat_test  = np.asarray(c_hat_test)
    c_train     = np.asarray(c_train)
    c_test      = np.asarray(c_test)
    y_train     = np.asarray(y_train).ravel()
    y_test      = np.asarray(y_test).ravel()

    N_tr, K, m = c_hat_train.shape
    N_te = c_hat_test.shape[0]

    X_train = c_hat_train.reshape(N_tr, K * m)
    X_test  = c_hat_test.reshape(N_te, K * m)

    # Step 1: Ridge removes all linearly concept-explained variance from embedding
    ridge1 = Ridge(alpha=alpha)
    ridge1.fit(c_train, X_train)
    R_train = X_train - ridge1.predict(c_train)
    R_test  = X_test  - ridge1.predict(c_test)

    # Step 2: Ridge(y → R); variance-weighted R² across K*m embedding dims
    ridge2 = Ridge(alpha=alpha)
    ridge2.fit(y_train.reshape(-1, 1), R_train)
    R_pred = ridge2.predict(y_test.reshape(-1, 1))
    r2    = float(r2_score(R_test, R_pred, multioutput="variance_weighted"))
    delta = max(0.0, r2)
    print(f"    CVL_global: r2={r2:.4f}  CVL={delta:.4f}")

    return {
        "CVL_global": delta,
        "r2":         r2,
    }


def _fit_mlp_regressor(X_train, Y_train, X_test, Y_test, hidden, max_iter):
    """
    Fit MLPRegressor(X → Y) and return (R_train, R_test) residuals in original scale.
    """
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    Xtr = scaler_x.fit_transform(X_train)
    Xte = scaler_x.transform(X_test)
    Ytr = scaler_y.fit_transform(Y_train)
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=0,
        alpha=1e-3,
    )
    mlp.fit(Xtr, Ytr)
    R_train = Y_train - scaler_y.inverse_transform(mlp.predict(Xtr))
    R_test  = Y_test  - scaler_y.inverse_transform(mlp.predict(Xte))
    return R_train, R_test


def _majority_baseline(y):
    """Accuracy of always predicting the most common class."""
    vals, counts = np.unique(y, return_counts=True)
    return float(counts.max() / len(y))


def _fit_mlp(X_train, y_train, X_test, y_test, hidden, max_iter):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=0,
        alpha=1e-3,
    )
    mlp.fit(X_tr, y_train)
    return float(mlp.score(X_te, y_test))


def compute_CVL_mlp(
    c_hat_train,
    c_hat_test,
    c_train,
    c_test,
    y_train,
    y_test,
    alpha=1.0,
    hidden=(64,),
    max_iter=500,
):
    """
    Per-concept CVL with an MLP probe instead of Ridge for step 2.

    Step 1 (same as compute_CVL): Ridge(c_k scalar → c_hat_k embedding) → residuals R_k
    Step 2 (new): MLPClassifier(R_k → y) accuracy vs majority-class baseline
    CVL_k = max(0, acc_k − baseline)   CVL = mean(CVL_k)

    Returns
    -------
    dict with keys:
        'CVL_mlp'              : float        — aggregate; near 0 = clean
        'CVL_mlp_per_concept'  : list[float]  — clipped delta-acc per concept
        'acc_per_concept'      : list[float]  — raw MLP accuracy per concept
        'baseline_acc'         : float        — majority-class accuracy
    """
    c_hat_train = np.asarray(c_hat_train)
    c_hat_test  = np.asarray(c_hat_test)
    c_train     = np.asarray(c_train)
    c_test      = np.asarray(c_test)
    y_train     = np.asarray(y_train).ravel()
    y_test      = np.asarray(y_test).ravel()

    K = c_hat_train.shape[1]
    baseline = _majority_baseline(y_test)

    cvl_per = []
    acc_per = []
    for k in range(K):
        # Step 1: remove linear concept-k variance from embedding-k
        ridge1 = Ridge(alpha=alpha)
        ridge1.fit(c_train[:, k].reshape(-1, 1), c_hat_train[:, k, :])
        R_train = c_hat_train[:, k, :] - ridge1.predict(c_train[:, k].reshape(-1, 1))
        R_test  = c_hat_test[:, k, :]  - ridge1.predict(c_test[:, k].reshape(-1, 1))

        # Step 2: MLP probe R → y
        acc = _fit_mlp(R_train, y_train, R_test, y_test, hidden, max_iter)
        delta = max(0.0, acc - baseline)
        print(f"      concept {k:3d}: acc={acc:.4f}  baseline={baseline:.4f}  Δ={delta:.4f}")
        acc_per.append(acc)
        cvl_per.append(delta)

    return {
        "CVL_mlp":             float(np.mean(cvl_per)),
        "CVL_mlp_per_concept": cvl_per,
        "acc_per_concept":     acc_per,
        "baseline_acc":        baseline,
    }


def compute_CVL_global_mlp(
    c_hat_train,
    c_hat_test,
    c_train,
    c_test,
    y_train,
    y_test,
    alpha=1.0,
    hidden=(128, 64),
    max_iter=500,
    pca_components=128,
):
    """
    Global CVL with an MLP probe instead of Ridge for step 2.

    Step 1 (same as compute_CVL_global): Ridge(c_true [N,K] → c_hat.reshape(N,K*m)) → residuals R
    Step 2 (new): optional PCA on R, then MLPClassifier(R → y) accuracy vs baseline
    CVL_global_mlp = max(0, acc − baseline)

    pca_components : int or None — reduce R to this many dims before MLP.
                     Recommended for large K*m (e.g. CUB: 3584-dim → 128).
                     Set None to skip PCA.

    Returns
    -------
    dict with keys:
        'CVL_global_mlp'     : float — clipped delta-acc; near 0 = clean
        'acc'                : float — raw MLP accuracy
        'baseline_acc'       : float — majority-class accuracy
        'pca_components_used': int or None
    """
    c_hat_train = np.asarray(c_hat_train)
    c_hat_test  = np.asarray(c_hat_test)
    c_train     = np.asarray(c_train)
    c_test      = np.asarray(c_test)
    y_train     = np.asarray(y_train).ravel()
    y_test      = np.asarray(y_test).ravel()

    N_tr, K, m = c_hat_train.shape
    N_te = c_hat_test.shape[0]

    X_train = c_hat_train.reshape(N_tr, K * m)
    X_test  = c_hat_test.reshape(N_te, K * m)

    # Step 1: remove linear concept variance from full embedding
    ridge1 = Ridge(alpha=alpha)
    ridge1.fit(c_train, X_train)
    R_train = X_train - ridge1.predict(c_train)
    R_test  = X_test  - ridge1.predict(c_test)

    # Optional PCA before MLP (important when K*m >> N)
    n_comp = None
    if pca_components is not None:
        n_comp = min(pca_components, R_train.shape[1], N_tr - 1)
        scaler = StandardScaler()
        R_train = scaler.fit_transform(R_train)
        R_test  = scaler.transform(R_test)
        pca = PCA(n_components=n_comp, random_state=0)
        R_train = pca.fit_transform(R_train)
        R_test  = pca.transform(R_test)
        print(f"    PCA: {K*m} → {n_comp} dims "
              f"(var explained={pca.explained_variance_ratio_.sum():.3f})")

    # Step 2: MLP probe R → y
    baseline = _majority_baseline(y_test)
    acc = _fit_mlp(R_train, y_train, R_test, y_test, hidden, max_iter)
    delta = max(0.0, acc - baseline)
    print(f"    CVL_global_mlp: acc={acc:.4f}  baseline={baseline:.4f}  Δ={delta:.4f}")

    return {
        "CVL_global_mlp":      delta,
        "acc":                 acc,
        "baseline_acc":        baseline,
        "pca_components_used": n_comp,
    }


def compute_CVL_mlp2(
    c_hat_train,
    c_hat_test,
    c_train,
    c_test,
    y_train,
    y_test,
    hidden_reg=(64,),
    hidden_cls=(64,),
    max_iter=500,
):
    """
    Per-concept CVL with MLP for both steps.

    Step 1: MLPRegressor(c_k scalar → c_hat_k embedding) → residuals R_k
            Removes non-linear concept-predictable variance from the embedding.
    Step 2: MLPClassifier(R_k → y) accuracy vs majority-class baseline.
    CVL_k = max(0, acc_k − baseline)   CVL = mean(CVL_k)

    Returns
    -------
    dict with keys:
        'CVL_mlp2'             : float
        'CVL_mlp2_per_concept' : list[float]
        'acc_per_concept'      : list[float]
        'baseline_acc'         : float
    """
    c_hat_train = np.asarray(c_hat_train)
    c_hat_test  = np.asarray(c_hat_test)
    c_train     = np.asarray(c_train)
    c_test      = np.asarray(c_test)
    y_train     = np.asarray(y_train).ravel()
    y_test      = np.asarray(y_test).ravel()

    K = c_hat_train.shape[1]
    baseline = _majority_baseline(y_test)

    cvl_per = []
    acc_per = []
    for k in range(K):
        c_tr_k = c_train[:, k].reshape(-1, 1)
        c_te_k = c_test[:, k].reshape(-1, 1)

        R_train, R_test = _fit_mlp_regressor(
            c_tr_k, c_hat_train[:, k, :],
            c_te_k, c_hat_test[:, k, :],
            hidden_reg, max_iter,
        )
        acc   = _fit_mlp(R_train, y_train, R_test, y_test, hidden_cls, max_iter)
        delta = max(0.0, acc - baseline)
        print(f"      concept {k:3d}: acc={acc:.4f}  baseline={baseline:.4f}  Δ={delta:.4f}")
        acc_per.append(acc)
        cvl_per.append(delta)

    return {
        "CVL_mlp2":             float(np.mean(cvl_per)),
        "CVL_mlp2_per_concept": cvl_per,
        "acc_per_concept":      acc_per,
        "baseline_acc":         baseline,
    }


def compute_CVL_global_mlp2(
    c_hat_train,
    c_hat_test,
    c_train,
    c_test,
    y_train,
    y_test,
    hidden_reg=(128, 64),
    hidden_cls=(128, 64),
    max_iter=500,
    pca_components=128,
):
    """
    Global CVL with MLP for both steps.

    Step 1: optional PCA to reduce K*m dims, then
            MLPRegressor(c_true [N,K] → PCA(c_hat) [N, n_comp]) → residuals R
    Step 2: MLPClassifier(R → y) accuracy vs baseline.
    CVL_global_mlp2 = max(0, acc − baseline)

    pca_components : int or None — applied to c_hat before step 1 to make the
                     regression tractable. None skips PCA (only feasible for small K*m).
    """
    c_hat_train = np.asarray(c_hat_train)
    c_hat_test  = np.asarray(c_hat_test)
    c_train     = np.asarray(c_train)
    c_test      = np.asarray(c_test)
    y_train     = np.asarray(y_train).ravel()
    y_test      = np.asarray(y_test).ravel()

    N_tr, K, m = c_hat_train.shape
    N_te = c_hat_test.shape[0]

    X_train = c_hat_train.reshape(N_tr, K * m)
    X_test  = c_hat_test.reshape(N_te, K * m)

    # Optional PCA on embeddings before step-1 regression (makes output tractable)
    n_comp = None
    if pca_components is not None:
        n_comp = min(pca_components, X_train.shape[1], N_tr - 1)
        scaler_emb = StandardScaler()
        X_train_s = scaler_emb.fit_transform(X_train)
        X_test_s  = scaler_emb.transform(X_test)
        pca = PCA(n_components=n_comp, random_state=0)
        X_train = pca.fit_transform(X_train_s)
        X_test  = pca.transform(X_test_s)
        print(f"    PCA: {K*m} → {n_comp} dims "
              f"(var explained={pca.explained_variance_ratio_.sum():.3f})")

    # Step 1: MLP regressor removes concept-predictable structure from (PCA'd) embedding
    R_train, R_test = _fit_mlp_regressor(
        c_train, X_train, c_test, X_test, hidden_reg, max_iter
    )

    # Step 2: MLP classifier detects task signal in residuals
    baseline = _majority_baseline(y_test)
    acc   = _fit_mlp(R_train, y_train, R_test, y_test, hidden_cls, max_iter)
    delta = max(0.0, acc - baseline)
    print(f"    CVL_global_mlp2: acc={acc:.4f}  baseline={baseline:.4f}  Δ={delta:.4f}")

    return {
        "CVL_global_mlp2":     delta,
        "acc":                 acc,
        "baseline_acc":        baseline,
        "pca_components_used": n_comp,
    }


def compute_ICVL(
    c_hat_train,
    c_hat_test,
    c_train,
    c_test,
    alpha=1.0,
    C=1.0,
    max_iter=500,
):
    """
    Inter-Concept Variance Leakage (ICVL).

    For every ordered pair (j, i) with j ≠ i:
      Step 1: Ridge(c_j scalar → ĉ_j embedding [N, m]) → residual R_j = ĉ_j − ĉ_j_hat
              Removes variance in j's embedding that is linearly explained by j's own
              ground-truth label.  R_j is orthogonal to c_j by construction.
      Step 2: Ridge(c_i → R_j [N, m]); variance-weighted R² across embedding dims.
              ICVL_ji = max(0, R²_ji)

    If c_i explains structure in R_j, concept j's embedding encodes information about
    concept i beyond what j's own label justifies — pure inter-concept leakage.

    Aggregate:
        ICVL = 1 / (K * (K-1)) * sum_{j != i} ICVL_ji

    Parameters
    ----------
    c_hat_train : array [N_train, K, m]
    c_hat_test  : array [N_test,  K, m]
    c_train     : array [N_train, K]   binary ground-truth concept labels
    c_test      : array [N_test,  K]
    alpha       : float  Ridge regularisation for both steps (default 1.0)
    C, max_iter : unused, kept for API compatibility

    Returns
    -------
    dict with keys:
        'ICVL'        : float        — scalar aggregate; near 0 = clean
        'ICVL_matrix' : list[list]   — K×K matrix of ICVL_ji; diagonal = 0
        'r2_matrix'   : list[list]   — K×K raw variance-weighted R²
    """
    c_hat_train = np.asarray(c_hat_train)
    c_hat_test  = np.asarray(c_hat_test)
    c_train     = np.asarray(c_train)
    c_test      = np.asarray(c_test)

    K = c_hat_train.shape[1]

    # Step 1 for every j: fit Ridge(c_j → ĉ_j) on train, compute residuals on both splits
    residuals_train = []
    residuals_test  = []
    for j in range(K):
        ridge = Ridge(alpha=alpha)
        ridge.fit(c_train[:, j].reshape(-1, 1), c_hat_train[:, j, :])
        R_tr = c_hat_train[:, j, :] - ridge.predict(c_train[:, j].reshape(-1, 1))
        R_te = c_hat_test[:, j, :]  - ridge.predict(c_test[:, j].reshape(-1, 1))
        residuals_train.append(R_tr)
        residuals_test.append(R_te)

    # Step 2 for every ordered pair (j, i), j ≠ i: Ridge(c_i → R_j), variance-weighted R²
    icvl_matrix = np.zeros((K, K))
    r2_matrix   = np.zeros((K, K))

    for j in range(K):
        for i in range(K):
            if i == j:
                continue
            ridge2 = Ridge(alpha=alpha)
            ridge2.fit(c_train[:, i].reshape(-1, 1), residuals_train[j])
            R_pred = ridge2.predict(c_test[:, i].reshape(-1, 1))
            r2    = float(r2_score(residuals_test[j], R_pred, multioutput="variance_weighted"))
            delta = max(0.0, r2)
            r2_matrix[j, i]   = r2
            icvl_matrix[j, i] = delta
            print(f"    ICVL ({j}→{i}): r2={r2:.4f}  ICVL={delta:.4f}")

    icvl_scalar = float(icvl_matrix.sum() / (K * (K - 1)))

    return {
        "ICVL":         icvl_scalar,
        "ICVL_matrix":  icvl_matrix.tolist(),
        "r2_matrix":    r2_matrix.tolist(),
    }
