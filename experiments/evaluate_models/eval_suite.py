"""
Shared evaluation utilities for the CEM/CRCEM test suite.

All functions accept numpy arrays so they are independent of the data loading
and model inference code, making them easy to reuse across datasets.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import joblib
import torch

from experiments.evaluate_model import (
    predict_c_y,
    run_multiple_interventions,
    compute_ois_nis_cas,
)
from xai_concept_leakage.models.construction import external_load_model_trainer
from xai_concept_leakage.metrics.mutual_information import (
    repeat_estimate_MI_concepts_task,
    repeat_estimate_MI_interconcept,
    compute_mi_matrix_parallel,
)
from xai_concept_leakage.metrics.leakage import (
    compute_CVL, compute_CVL_global,
    compute_CVL_mlp, compute_CVL_global_mlp,
    compute_CVL_mlp2, compute_CVL_global_mlp2,
    compute_ICVL,
)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _to_np(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def load_predictions(model_path, x2c_extractor, dl):
    """
    Run CEM inference on dl and return a dict of numpy arrays.

    Keys
    ----
    c_prob   : [N, K]      soft concept probabilities
    c_pred   : [N, K]      hard concept predictions (thresholded at 0.5)
    c_true   : [N, K]      ground-truth concept labels
    y_pred   : [N, C]      task logits
    y_true   : [N]         ground-truth task labels
    c_pos    : [N, K, m]   positive concept embeddings
    c_neg    : [N, K, m]   negative concept embeddings
    c_mix    : [N, K, m]   weighted mix embedding (feeds task head)
    n_concepts, emb_size   : ints derived from shapes
    """
    out = predict_c_y(
        dl=dl,
        model_path=model_path,
        x2c_extractor=x2c_extractor,
        c_sem_out=True,
        soft_prob_out=True,
        vec_emb_out=True,
    )
    c_sem, c_prob_t, c_true_t, y_pred_t, y_true_t, c_pos_t, c_neg_t = out

    c_prob   = _to_np(c_prob_t)
    c_true   = _to_np(c_true_t)
    y_pred   = _to_np(y_pred_t)
    y_true   = _to_np(y_true_t).ravel()
    c_pos    = _to_np(c_pos_t)
    c_neg    = _to_np(c_neg_t)

    n_concepts = c_true.shape[1]
    emb_size   = c_sem.shape[1] // n_concepts
    c_mix      = _to_np(c_sem).reshape(-1, n_concepts, emb_size)

    return {
        "c_prob":      c_prob,
        "c_pred":      (c_prob > 0.5).astype(float),
        "c_true":      c_true,
        "y_pred":      y_pred,
        "y_true":      y_true,
        "c_pos":       c_pos,
        "c_neg":       c_neg,
        "c_mix":       c_mix,
        "n_concepts":  n_concepts,
        "emb_size":    emb_size,
    }


# ---------------------------------------------------------------------------
# Error-set analysis
# ---------------------------------------------------------------------------

def error_set_analysis(c_pred, c_true, y_pred, y_true):
    """
    Bucket test samples by how many concepts are predicted incorrectly,
    and report task accuracy per bucket.

    Parameters
    ----------
    c_pred : [N, K]  hard concept predictions (0/1)
    c_true : [N, K]  ground-truth concept labels
    y_pred : [N, C]  task logits (argmax taken internally)
    y_true : [N]     ground-truth task labels

    Returns
    -------
    dict  {n_wrong: {"n_samples": int, "task_acc": float}}
    ordered by n_wrong ascending.
    """
    n_wrong   = (c_pred != c_true).sum(axis=1).astype(int)   # [N]
    y_correct = (y_pred.argmax(axis=-1) == y_true)            # [N] bool

    result = {}
    for k in sorted(np.unique(n_wrong)):
        mask = (n_wrong == k)
        result[int(k)] = {
            "n_samples": int(mask.sum()),
            "task_acc":  float(y_correct[mask].mean()),
        }
    return result


def print_error_set(analysis, label=""):
    header = f"Error-set analysis {label}"
    print(f"\n{header}")
    print(f"  {'# wrong':>8}  {'N':>7}  {'task acc':>9}")
    print(f"  {'-'*30}")
    for k, v in analysis.items():
        print(f"  {k:>8}  {v['n_samples']:>7}  {v['task_acc']:>9.4f}")


# ---------------------------------------------------------------------------
# MI leakage metrics
# ---------------------------------------------------------------------------

def compute_ctl(c_mix_flat, y_true, n_concepts, repeats=3, n_neighbors=3):
    """
    Concepts-Task Leakage (CTL): normalised MI between each concept embedding
    and the task label.

    c_mix_flat : [N, K*m]  flat weighted-mix embeddings
    Returns    : (mean [K], se [K]) if repeats > 1 else mean [K]
    """
    return repeat_estimate_MI_concepts_task(
        c_mix_flat, y_true,
        n_concepts=n_concepts,
        repeats=repeats,
        n_neighbors=n_neighbors,
        normalise=True,
        return_avg=True,
    )


def compute_icl(c_mix_flat, n_concepts, repeats=3, n_neighbors=3, n_jobs=-1):
    """
    Inter-Concept Leakage (ICL): normalised pairwise MI between concept embeddings.

    Uses compute_mi_matrix_parallel for parallelised computation across pairs.
    Returns: (mean scalar, se scalar) — mean over all off-diagonal pairs.

    n_jobs: number of parallel workers (-1 = all cores).
    """
    import os
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    run_means = []
    for _ in range(repeats):
        tril = compute_mi_matrix_parallel(
            c_mix_flat,
            n_neighbors=n_neighbors,
            normalise=True,
            flatten=True,
            n_jobs=n_jobs,
        )
        run_means.append(float(np.mean(tril)))
    return float(np.mean(run_means)), float(np.std(run_means) / max(1, np.sqrt(len(run_means) - 1)))


# ---------------------------------------------------------------------------
# Probe accuracy vs PCA components
# ---------------------------------------------------------------------------

def _probe_one_concept(k, X_tr_k, X_te_k, y_train, y_test, max_components):
    """PCA + LR probe for one concept slot; returns accuracy array [max_components]."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_k)
    X_te_s = scaler.transform(X_te_k)

    pca = PCA(n_components=max_components, random_state=0)
    X_tr_pca = pca.fit_transform(X_tr_s)
    X_te_pca = pca.transform(X_te_s)

    acc_k = np.zeros(max_components)
    for n in range(1, max_components + 1):
        lr = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs",
                                multi_class="auto", random_state=0)
        lr.fit(X_tr_pca[:, :n], y_train)
        acc_k[n - 1] = lr.score(X_te_pca[:, :n], y_test)
    return k, acc_k


def probe_accuracy_vs_pca(c_mix_train, c_mix_test, y_train, y_test,
                          max_components=None, n_jobs=-1):
    """
    For each concept k, project c_mix[:, k, :] to 1..max_components PCA dims
    and fit a logistic regression to predict the task label y.

    Parameters
    ----------
    c_mix_train : [N_train, K, m]
    c_mix_test  : [N_test,  K, m]
    y_train     : [N_train]
    y_test      : [N_test]
    max_components : int, capped at emb_size (default = emb_size)
    n_jobs     : parallelism over K concepts (prefer=threads to avoid spawn)

    Returns
    -------
    acc : ndarray [K, max_components]  test accuracy per concept per n_components
    """
    K, m = c_mix_train.shape[1], c_mix_train.shape[2]
    max_components = min(max_components or m, m, c_mix_train.shape[0] - 1)

    results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        delayed(_probe_one_concept)(
            k, c_mix_train[:, k, :], c_mix_test[:, k, :],
            y_train, y_test, max_components
        )
        for k in range(K)
    )

    acc = np.zeros((K, max_components))
    for k, acc_k in results:
        acc[k] = acc_k
    return acc   # [K, max_components]


# ---------------------------------------------------------------------------
# PCA scatter plot (one concept, coloured by y)
# ---------------------------------------------------------------------------

def pca_scatter_concept(c_hat, y, concept_idx, title="", save_path=None):
    """
    2-D PCA scatter of c_hat[:, concept_idx, :] coloured by task label y.

    c_hat : [N, K, m]
    y     : [N]
    """
    X = c_hat[:, concept_idx, :]
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X_s)

    classes = np.unique(y)
    cmap = plt.cm.get_cmap("tab10", len(classes))

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, cls in enumerate(classes):
        mask = (y == cls)
        ax.scatter(Z[mask, 0], Z[mask, 1], s=6, alpha=0.5,
                   color=cmap(i), label=str(cls))
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
    ax.set_title(title or f"Concept {concept_idx} embeddings coloured by y")
    if len(classes) <= 10:
        ax.legend(title="y", markerscale=2, fontsize=7, loc="best")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved PCA scatter → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Prototype alignment plot and scalar metrics
# ---------------------------------------------------------------------------

def prototype_alignment_score(c_pos, c_neg):
    """
    Scalar metrics for how well pos/neg prototypes are separated.

    c_pos, c_neg : [N, K, m]

    Returns
    -------
    pos_neg_dist : mean Euclidean distance between pos_k and neg_k centroids
    concept_sep  : mean pairwise distance between per-concept centroids
                   (centroid = (pos_k + neg_k) / 2)
    """
    pos_c = c_pos.mean(axis=0)          # [K, m]
    neg_c = c_neg.mean(axis=0)
    K = pos_c.shape[0]

    pos_neg_dist = float(np.mean([np.linalg.norm(pos_c[k] - neg_c[k]) for k in range(K)]))

    concept_centroid = (pos_c + neg_c) / 2
    dists = [np.linalg.norm(concept_centroid[i] - concept_centroid[j])
             for i in range(K) for j in range(i + 1, K)]
    concept_sep = float(np.mean(dists)) if dists else 0.0

    return {"pos_neg_dist": pos_neg_dist, "concept_sep": concept_sep}


def prototype_alignment_plot(c_pos_a, c_neg_a, c_pos_b, c_neg_b,
                              label_a="CEM", label_b="CRCEM",
                              concept_names=None, title="", save_path=None):
    """
    Side-by-side PCA of per-concept pos/neg prototype centroids for two models.

    Centroids are computed across all N samples then projected jointly so both
    models share the same PCA axes.  Filled marker = positive prototype,
    hollow marker = negative prototype.  Lines connect pos↔neg of the same concept.

    c_pos_a, c_neg_a : [N, K, m]  positive/negative embeddings for model A
    c_pos_b, c_neg_b : [N, K, m]  positive/negative embeddings for model B

    Returns scalar metrics dict (pos_neg_dist and concept_sep for each model).
    """
    K = c_pos_a.shape[1]
    concept_labels = concept_names or [str(k) for k in range(K)]

    pos_a = c_pos_a.mean(axis=0)   # [K, m]
    neg_a = c_neg_a.mean(axis=0)
    pos_b = c_pos_b.mean(axis=0)
    neg_b = c_neg_b.mean(axis=0)

    scores_a = prototype_alignment_score(c_pos_a, c_neg_a)
    scores_b = prototype_alignment_score(c_pos_b, c_neg_b)

    # Joint PCA so both models share the same 2-D axes
    all_centroids = np.concatenate([pos_a, neg_a, pos_b, neg_b], axis=0)  # [4K, m]
    scaler = StandardScaler()
    all_2d = PCA(n_components=2, random_state=0).fit_transform(
        scaler.fit_transform(all_centroids)
    )
    pos_a_2d, neg_a_2d = all_2d[:K], all_2d[K:2 * K]
    pos_b_2d, neg_b_2d = all_2d[2 * K:3 * K], all_2d[3 * K:]

    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(K)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    for ax, pos_2d, neg_2d, lbl, scores in [
        (axes[0], pos_a_2d, neg_a_2d, label_a, scores_a),
        (axes[1], pos_b_2d, neg_b_2d, label_b, scores_b),
    ]:
        for k in range(K):
            color = cmap(k)
            ax.scatter(pos_2d[k, 0], pos_2d[k, 1], s=140, color=color,
                       marker="o", zorder=3, label=concept_labels[k])
            ax.scatter(neg_2d[k, 0], neg_2d[k, 1], s=140, color=color,
                       marker="o", facecolors="none", linewidths=2, zorder=3)
            ax.plot([pos_2d[k, 0], neg_2d[k, 0]], [pos_2d[k, 1], neg_2d[k, 1]],
                    color=color, alpha=0.5, linewidth=1.5)
        ax.set_title(
            f"{lbl}\npos-neg dist={scores['pos_neg_dist']:.3f}"
            f"  concept sep={scores['concept_sep']:.3f}"
        )
        ax.set_xlabel("PC1")

    axes[0].set_ylabel("PC2")
    axes[0].legend(title="Concept (●=pos, ○=neg)", markerscale=1.2,
                   fontsize=8, loc="best")
    fig.suptitle(title or "Prototype alignment")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved prototype alignment plot → {save_path}")
    plt.close(fig)

    return {
        label_a: scores_a,
        label_b: scores_b,
    }


# ---------------------------------------------------------------------------
# Probe accuracy curve plot (one or averaged across concepts)
# ---------------------------------------------------------------------------

def plot_probe_curve(acc_cem, acc_crcem, title="", save_path=None, average=False):
    """
    Plot probe accuracy vs n_PCA_components for CEM and CRCEM.

    acc_cem, acc_crcem : [K, max_comp]  or [max_comp] if already averaged
    average            : if True, average over K concepts before plotting
    """
    if average or acc_cem.ndim == 2:
        mu_cem   = acc_cem.mean(axis=0)   if acc_cem.ndim   == 2 else acc_cem
        mu_crcem = acc_crcem.mean(axis=0) if acc_crcem.ndim == 2 else acc_crcem
        se_cem   = acc_cem.std(axis=0)  / np.sqrt(acc_cem.shape[0])   if acc_cem.ndim   == 2 else None
        se_crcem = acc_crcem.std(axis=0) / np.sqrt(acc_crcem.shape[0]) if acc_crcem.ndim == 2 else None
    else:
        mu_cem, mu_crcem = acc_cem, acc_crcem
        se_cem = se_crcem = None

    xs = np.arange(1, len(mu_cem) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, mu_cem,   label="CEM",   color="tab:blue")
    ax.plot(xs, mu_crcem, label="CRCEM", color="tab:orange")
    if se_cem is not None:
        ax.fill_between(xs, mu_cem   - se_cem,   mu_cem   + se_cem,   alpha=0.2, color="tab:blue")
        ax.fill_between(xs, mu_crcem - se_crcem, mu_crcem + se_crcem, alpha=0.2, color="tab:orange")
    ax.set_xlabel("PCA components")
    ax.set_ylabel("Probe accuracy (task)")
    ax.set_title(title or "Probe accuracy vs PCA components")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved probe curve → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scalar-vs-embedding CVL probe
# ---------------------------------------------------------------------------

def embedding_probe_delta(c_prob_train, c_prob_test,
                          c_mix_train, c_mix_test,
                          y_train, y_test):
    """
    ΔAcc = A_emb − A_scalar, where:
      A_scalar = LR accuracy from full prob vector [ĉ_1,...,ĉ_K] → y
      A_emb    = LR accuracy from flattened embedding [c_mix_1,...,c_mix_K] → y

    Positive delta means the embeddings collectively carry more task-predictive
    information than the concept probabilities alone.

    Returns
    -------
    dict with:
        'acc_scalar' : float
        'acc_emb'    : float
        'delta'      : float  A_emb − A_scalar
    """
    scaler = StandardScaler()

    X_prob_tr = scaler.fit_transform(c_prob_train)
    X_prob_te = scaler.transform(c_prob_test)
    lr_scalar = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                   multi_class="auto", random_state=0)
    lr_scalar.fit(X_prob_tr, y_train)
    acc_scalar = float(lr_scalar.score(X_prob_te, y_test))

    N_tr, K, m = c_mix_train.shape
    X_emb_tr = scaler.fit_transform(c_mix_train.reshape(N_tr, K * m))
    X_emb_te = scaler.transform(c_mix_test.reshape(len(y_test), K * m))
    lr_emb = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                multi_class="auto", random_state=0)
    lr_emb.fit(X_emb_tr, y_train)
    acc_emb = float(lr_emb.score(X_emb_te, y_test))

    return {
        "acc_scalar": acc_scalar,
        "acc_emb":    acc_emb,
        "delta":      acc_emb - acc_scalar,
    }


# ---------------------------------------------------------------------------
# Task leakage probe  y|c  vs  y|[c, Ĉ]
# ---------------------------------------------------------------------------

def task_leakage_probe(c_true_train, c_true_test,
                       c_mix_train, c_mix_test,
                       y_train, y_test):
    """
    Measure how much extra task-relevant information the embeddings carry
    beyond what ground-truth concept labels already explain.

    Two logistic regression probes:
      Probe 1: c_true              → y   (GT concepts only)
      Probe 2: [c_true, flat(Ĉ)]  → y   (GT concepts + embeddings)

    Returns
    -------
    dict with:
        'acc_concepts'   : float  accuracy of probe 1
        'acc_concat'     : float  accuracy of probe 2
        'acc_delta'      : float  acc_concat − acc_concepts
        'bce_concepts'   : float  mean BCE of probe 1
        'bce_concat'     : float  mean BCE of probe 2
        'bce_delta'      : float  bce_concat − bce_concepts  (negative = embeddings help)
    """
    from sklearn.metrics import log_loss

    N_tr, K, m = c_mix_train.shape
    N_te = len(y_test)

    scaler_c   = StandardScaler()
    scaler_emb = StandardScaler()

    X_c_tr  = scaler_c.fit_transform(c_true_train.astype(float))
    X_c_te  = scaler_c.transform(c_true_test.astype(float))
    X_emb_tr = scaler_emb.fit_transform(c_mix_train.reshape(N_tr, K * m))
    X_emb_te = scaler_emb.transform(c_mix_test.reshape(N_te, K * m))

    X_cat_tr = np.concatenate([X_c_tr, X_emb_tr], axis=1)
    X_cat_te = np.concatenate([X_c_te, X_emb_te], axis=1)

    n_classes = len(np.unique(y_train))
    solver = "lbfgs"
    lr_c = LogisticRegression(max_iter=1000, C=1.0, solver=solver,
                              multi_class="auto", random_state=0)
    lr_c.fit(X_c_tr, y_train)

    lr_cat = LogisticRegression(max_iter=1000, C=1.0, solver=solver,
                                multi_class="auto", random_state=0)
    lr_cat.fit(X_cat_tr, y_train)

    acc_c   = float(lr_c.score(X_c_te, y_test))
    acc_cat = float(lr_cat.score(X_cat_te, y_test))

    prob_c   = lr_c.predict_proba(X_c_te)
    prob_cat = lr_cat.predict_proba(X_cat_te)
    bce_c   = float(log_loss(y_test, prob_c))
    bce_cat = float(log_loss(y_test, prob_cat))

    return {
        "acc_concepts": acc_c,
        "acc_concat":   acc_cat,
        "acc_delta":    acc_cat - acc_c,
        "bce_concepts": bce_c,
        "bce_concat":   bce_cat,
        "bce_delta":    bce_cat - bce_c,
    }


# ---------------------------------------------------------------------------
# Intervention analysis
# ---------------------------------------------------------------------------

def run_intervention_curve(ckpt, x2c_extractor, train_dl, val_dl, test_dl,
                           policies=("random", "coop"), repeats=3):
    """
    Sweep intervention budgets k=0..K with the given policies (competence=1).

    Returns
    -------
    dict  {policy: list-of-arrays}  where each array is task_acc at each k
    """
    return run_multiple_interventions(
        model_path=ckpt,
        x2c_extractor=x2c_extractor,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        competence_levels=[1],
        policies=list(policies),
        repeats=repeats,
    )


def adversarial_concept_flip(ckpt, x2c_extractor, train_dl,
                             test_preds, repeats=3, seed=0):
    """
    Adversarial intervention curve: sweep k=0..K concepts flipped from correct
    to wrong, measuring task accuracy at each budget.

    At each budget k, for each sample we randomly select k concepts from those
    currently predicted correctly and flip their value (prob → 1-c_true).
    Concepts already predicted incorrectly are never touched.

    Returns
    -------
    dict with:
        'acc_curve'  : [K+1]  mean task accuracy at each flip budget (avg over repeats)
        'acc_0'      : float  baseline (k=0, no flips)
        'acc_all'    : float  all correct concepts flipped
    """
    model, _, _ = external_load_model_trainer(
        train_dl, ckpt, x2c_extractor, output_config=True
    )
    model.eval()

    c_pred  = test_preds["c_pred"].astype(np.float32)   # [N, K]
    c_true  = test_preds["c_true"].astype(np.float32)   # [N, K]
    c_pos   = test_preds["c_pos"]                        # [N, K, m]
    c_neg   = test_preds["c_neg"]                        # [N, K, m]
    y_true  = test_preds["y_true"]                       # [N]

    pos_t = torch.tensor(c_pos, dtype=torch.float32)
    neg_t = torch.tensor(c_neg, dtype=torch.float32)
    correct = (c_pred == c_true)                         # [N, K] bool
    K = c_pred.shape[1]

    def _acc_at_k(k, rng):
        prob = c_pred.copy()
        for i in range(len(y_true)):
            correct_idxs = np.where(correct[i])[0]
            if k > 0 and len(correct_idxs) > 0:
                flip_idxs = rng.choice(correct_idxs,
                                       size=min(k, len(correct_idxs)),
                                       replace=False)
                prob[i, flip_idxs] = 1.0 - c_true[i, flip_idxs]
        p = torch.tensor(prob, dtype=torch.float32).unsqueeze(-1)
        c_mix = pos_t * p + neg_t * (1 - p)
        N, K_, m = c_mix.shape
        with torch.no_grad():
            y_logits = model.c2y_model(c_mix.view(N, K_ * m))
        return float((y_logits.argmax(-1).numpy() == y_true).mean())

    acc_curves = []
    for r in range(repeats):
        rng = np.random.default_rng(seed + r)
        curve = [_acc_at_k(k, rng) for k in range(K + 1)]
        acc_curves.append(curve)

    mean_curve = np.mean(acc_curves, axis=0)
    return {
        "acc_curve": mean_curve.tolist(),
        "acc_0":     float(mean_curve[0]),
        "acc_all":   float(mean_curve[-1]),
    }


# ---------------------------------------------------------------------------
# CVL wrapper (TabularToy / dSprites only)
# ---------------------------------------------------------------------------

def run_icvl(c_mix_train, c_mix_test, c_true_train, c_true_test):
    """
    Inter-Concept Variance Leakage (ICVL).

    Step 1 per concept j: Ridge(c_j → ĉ_j embedding) → residuals R_j
    Step 2 per pair (j,i): LogisticRegression(R_j → c_i binary) → Δacc over baseline

    Returns compute_ICVL result dict.
    """
    return compute_ICVL(
        c_hat_train=c_mix_train,
        c_hat_test=c_mix_test,
        c_train=c_true_train,
        c_test=c_true_test,
    )


def run_cvl(c_mix_train, c_mix_test, c_true_train, c_true_test, y_train, y_test):
    """Thin wrapper around compute_CVL that takes [N, K, m] arrays directly."""
    return compute_CVL(
        c_hat_train=c_mix_train,
        c_hat_test=c_mix_test,
        c_train=c_true_train,
        c_test=c_true_test,
        y_train=y_train,
        y_test=y_test,
    )


def run_cvl_global(c_mix_train, c_mix_test, c_true_train, c_true_test, y_train, y_test):
    """Thin wrapper around compute_CVL_global that takes [N, K, m] arrays directly."""
    return compute_CVL_global(
        c_hat_train=c_mix_train,
        c_hat_test=c_mix_test,
        c_train=c_true_train,
        c_test=c_true_test,
        y_train=y_train,
        y_test=y_test,
    )


def run_cvl_mlp(c_mix_train, c_mix_test, c_true_train, c_true_test, y_train, y_test,
                hidden=(64,), max_iter=500):
    """Per-concept CVL with MLP probe in step 2."""
    return compute_CVL_mlp(
        c_hat_train=c_mix_train, c_hat_test=c_mix_test,
        c_train=c_true_train, c_test=c_true_test,
        y_train=y_train, y_test=y_test,
        hidden=hidden, max_iter=max_iter,
    )


def run_cvl_global_mlp(c_mix_train, c_mix_test, c_true_train, c_true_test, y_train, y_test,
                       hidden=(128, 64), max_iter=500, pca_components=128):
    """Global CVL with PCA + MLP probe in step 2."""
    return compute_CVL_global_mlp(
        c_hat_train=c_mix_train, c_hat_test=c_mix_test,
        c_train=c_true_train, c_test=c_true_test,
        y_train=y_train, y_test=y_test,
        hidden=hidden, max_iter=max_iter, pca_components=pca_components,
    )


def run_cvl_mlp2(c_mix_train, c_mix_test, c_true_train, c_true_test, y_train, y_test,
                 hidden_reg=(64,), hidden_cls=(64,), max_iter=500):
    """Per-concept CVL with MLP for both steps."""
    return compute_CVL_mlp2(
        c_hat_train=c_mix_train, c_hat_test=c_mix_test,
        c_train=c_true_train, c_test=c_true_test,
        y_train=y_train, y_test=y_test,
        hidden_reg=hidden_reg, hidden_cls=hidden_cls, max_iter=max_iter,
    )


def run_cvl_global_mlp2(c_mix_train, c_mix_test, c_true_train, c_true_test, y_train, y_test,
                        hidden_reg=(128, 64), hidden_cls=(128, 64), max_iter=500,
                        pca_components=128):
    """Global CVL with MLP for both steps (PCA applied before step 1)."""
    return compute_CVL_global_mlp2(
        c_hat_train=c_mix_train, c_hat_test=c_mix_test,
        c_train=c_true_train, c_test=c_true_test,
        y_train=y_train, y_test=y_test,
        hidden_reg=hidden_reg, hidden_cls=hidden_cls,
        max_iter=max_iter, pca_components=pca_components,
    )


# ---------------------------------------------------------------------------
# OIS / NIS / CAS wrappers
# ---------------------------------------------------------------------------

def run_ois_nis_cas(ckpt, x2c_extractor, train_dl, test_dl, repeats=1):
    """
    Compute OIS, NIS, and CAS for a single checkpoint.

    These metrics go through evaluate_representation_metrics internally, which
    loads the model from the checkpoint path and runs its own inference —
    they do not consume the numpy prediction dicts from load_predictions.

    Returns
    -------
    dict with scalar floats: {'ois': float, 'nis': float, 'cas': float}
    """
    raw = compute_ois_nis_cas(
        model_path=ckpt,
        dl=test_dl,
        x2c_extractor=x2c_extractor,
        train_dl=train_dl,
        run_ois=True,
        run_nis=True,
        run_cas=True,
        repeats=repeats,
        rerun=True,
    )
    return {
        "ois": float(np.mean(raw["ois"])),
        "nis": float(np.mean(raw["nis"])),
        "cas": float(np.mean(raw["cas"])),
    }


# ---------------------------------------------------------------------------
# dSprites-specific leakage analysis
# ---------------------------------------------------------------------------

def dsprites_error_set_analysis(c_pred, c_true, y_pred, y_true):
    """
    Hard leakage test exploiting the bijective task structure of dSprites:

        y = (1-c1)(2*c4 + c5 + 4) + c1*(2*c2 + c3)

    Within each branch (c1=0 or c1=1), the relevant concept pair maps
    bijectively to y, so a correct task prediction with a wrong relevant
    concept is impossible through legitimate concept use.

    Concept indexing (0-based): c1=idx0, c2=idx1, c3=idx2, c4=idx3, c5=idx4

    Filter: only samples where c1 is predicted correctly.

    Groups
    ------
    Leakage set  : c1 correct AND >=1 relevant concept wrong
    Sanity set   : c1 correct AND 0 relevant wrong AND >=1 irrelevant wrong

    Returns
    -------
    dict with leakage_rate, sanity_check_rate, n_relevant_errors,
    n_irrelevant_errors, leakage_rate_square, leakage_rate_nonsquare
    """
    c_pred  = np.asarray(c_pred)
    c_true  = np.asarray(c_true)
    y_pred  = np.asarray(y_pred)
    y_true  = np.asarray(y_true)

    y_hat = y_pred.argmax(-1) if y_pred.ndim == 2 else y_pred

    # concept indices
    IDX_C1, IDX_C2, IDX_C3, IDX_C4, IDX_C5 = 0, 1, 2, 3, 4

    # c1 is always relevant — a wrong c1 is immediately a relevant error.
    # Branch is determined by the TRUE c1 (the ground truth structure).
    # Within each branch, the branch-specific relevant concepts also apply.
    BRANCH_RELEVANT = {
        0: [IDX_C4, IDX_C5],   # non-square
        1: [IDX_C2, IDX_C3],   # square
    }
    BRANCH_IRRELEVANT = {
        0: [IDX_C2, IDX_C3],
        1: [IDX_C4, IDX_C5],
    }

    c1_wrong = c_pred[:, IDX_C1] != c_true[:, IDX_C1]

    def _branch_results(branch_val):
        # True branch membership (by GT c1)
        in_branch = c_true[:, IDX_C1] == branch_val
        rel   = BRANCH_RELEVANT[branch_val]
        irrel = BRANCH_IRRELEVANT[branch_val]

        # relevant wrong = c1 wrong OR branch-specific relevant concept wrong
        branch_rel_wrong = np.any(
            c_pred[in_branch][:, rel] != c_true[in_branch][:, rel], axis=1
        )
        rel_wrong_branch  = c1_wrong[in_branch] | branch_rel_wrong
        irrel_wrong_branch = np.any(
            c_pred[in_branch][:, irrel] != c_true[in_branch][:, irrel], axis=1
        )

        leak_mask   = rel_wrong_branch
        sanity_mask = (~rel_wrong_branch) & irrel_wrong_branch

        idx = np.where(in_branch)[0]
        leak_acc   = float((y_hat[idx[leak_mask]]   == y_true[idx[leak_mask]]).mean())   if leak_mask.any()   else float("nan")
        sanity_acc = float((y_hat[idx[sanity_mask]] == y_true[idx[sanity_mask]]).mean()) if sanity_mask.any() else float("nan")
        return leak_acc, sanity_acc, int(leak_mask.sum()), int(sanity_mask.sum())

    lr_sq,  sr_sq,  n_rel_sq,  n_irrel_sq  = _branch_results(1)
    lr_nsq, sr_nsq, n_rel_nsq, n_irrel_nsq = _branch_results(0)

    # Combined across both branches
    rel_wrong_all   = np.zeros(len(y_true), dtype=bool)
    irrel_wrong_all = np.zeros(len(y_true), dtype=bool)
    for branch_val in [0, 1]:
        in_branch = c_true[:, IDX_C1] == branch_val
        rel   = BRANCH_RELEVANT[branch_val]
        irrel = BRANCH_IRRELEVANT[branch_val]
        idx   = np.where(in_branch)[0]
        branch_rel_wrong = np.any(
            c_pred[in_branch][:, rel] != c_true[in_branch][:, rel], axis=1
        )
        rw  = c1_wrong[in_branch] | branch_rel_wrong
        irw = np.any(c_pred[in_branch][:, irrel] != c_true[in_branch][:, irrel], axis=1)
        rel_wrong_all[idx[rw]]          = True
        irrel_wrong_all[idx[irw & ~rw]] = True

    leak_rate   = float((y_hat[rel_wrong_all]   == y_true[rel_wrong_all]).mean())   if rel_wrong_all.any()   else float("nan")
    sanity_rate = float((y_hat[irrel_wrong_all] == y_true[irrel_wrong_all]).mean()) if irrel_wrong_all.any() else float("nan")

    return {
        "leakage_rate":           leak_rate,
        "sanity_check_rate":      sanity_rate,
        "n_relevant_errors":      int(rel_wrong_all.sum()),
        "n_irrelevant_errors":    int(irrel_wrong_all.sum()),
        "leakage_rate_square":    lr_sq,
        "leakage_rate_nonsquare": lr_nsq,
        "n_relevant_square":      n_rel_sq,
        "n_relevant_nonsquare":   n_rel_nsq,
    }


def print_dsprites_leakage(result, label=""):
    print(f"\n  dSprites leakage analysis {label}")
    print(f"    Leakage rate  (relevant wrong):  {result['leakage_rate']:.4f}  (n={result['n_relevant_errors']})")
    print(f"    Sanity rate   (irrelevant wrong): {result['sanity_check_rate']:.4f}  (n={result['n_irrelevant_errors']})")
    print(f"    Square branch:     {result['leakage_rate_square']:.4f}  (n={result['n_relevant_square']})")
    print(f"    Non-square branch: {result['leakage_rate_nonsquare']:.4f}  (n={result['n_relevant_nonsquare']})")


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

ALL_FOLDS = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]


def find_checkpoints(folder, name_filter, lam_c_filter, folds):
    """
    Return {fold_name: absolute_checkpoint_path} for all requested folds.

    Scans `folder` for model sub-directories whose name contains both
    `name_filter` and `lam_c_filter`, then finds the .pt file for each fold.
    Raises AssertionError if no matching model directory is found.
    """
    model_dir = None
    for d in sorted(os.listdir(folder)):
        if not os.path.isdir(os.path.join(folder, d)) or d == "auto":
            continue
        if d.startswith(name_filter) and lam_c_filter in d:
            model_dir = d
            break
    assert model_dir, (
        f"No model dir matching '{name_filter}' + '{lam_c_filter}' in {folder}"
    )

    checkpoints = {}
    model_path = os.path.join(folder, model_dir)
    pt_files = sorted(f for f in os.listdir(model_path) if f.endswith(".pt"))
    for fold in folds:
        matches = [f for f in pt_files if fold in f]
        if matches:
            checkpoints[fold] = os.path.join(model_path, matches[0])
        else:
            print(f"  WARNING: no checkpoint found for {fold} in {model_dir}")
    return checkpoints


# ---------------------------------------------------------------------------
# Inter-concept leakage probe
# ---------------------------------------------------------------------------

def _fit_eval_probe(X_tr, X_te, y_tr, y_te):
    """
    Fit a balanced logistic regression probe and return (accuracy, bce_loss) on test.
    Falls back gracefully if only one class is present in training.
    """
    lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=0
    )
    try:
        lr.fit(X_tr, y_tr)
        acc = float(lr.score(X_te, y_te))
        if 1 in lr.classes_:
            pos_col = list(lr.classes_).index(1)
            p = np.clip(lr.predict_proba(X_te)[:, pos_col], 1e-7, 1 - 1e-7)
        else:
            p = np.full(len(y_te), 1e-7)
        bce = float(-np.mean(y_te * np.log(p) + (1 - y_te) * np.log(1 - p)))
    except Exception:
        acc = 0.5
        bce = float(np.log(2))
    return acc, bce


def _probe_row(i, K, X_tr_i, X_te_i, c_true_train, c_true_test):
    """Fit K-1 probes with concept i as predictor; return (i, acc_row, loss_row)."""
    acc_row  = np.ones(K)
    loss_row = np.zeros(K)
    for j in range(K):
        if i == j:
            continue
        acc_row[j], loss_row[j] = _fit_eval_probe(
            X_tr_i, X_te_i, c_true_train[:, j], c_true_test[:, j]
        )
    return i, acc_row, loss_row


def interconcept_probe_analysis(
    c_mix_train,
    c_mix_test,
    c_true_train,
    c_true_test,
    baseline_cache_path=None,
    n_jobs=-1,
):
    """
    Measure inter-concept leakage via linear probes for every concept pair (i, j).

    Baseline probe : c_i (GT scalar)       → c_j
    Full probe     : [ĉ_i (mix emb), c_i] → c_j

    The difference isolates extra information about c_j encoded in ĉ_i beyond
    what the ground-truth c_i already provides.

    Parameters
    ----------
    c_mix_train      : [N_train, K, m]
    c_mix_test       : [N_test,  K, m]
    c_true_train     : [N_train, K]   binary GT labels
    c_true_test      : [N_test,  K]
    baseline_cache_path : str or None — path to cache the GT-only baseline matrices.
                          Cached because the baseline is model-independent.
    n_jobs           : int — number of parallel workers (-1 = all CPUs)

    Returns
    -------
    dict with K×K numpy arrays:
        acc_baseline, acc_concat, acc_diff
        loss_baseline, loss_concat, loss_diff, loss_diff_norm
    Diagonal is 1 for acc matrices, 0 for loss/diff matrices.
    """
    K = c_true_train.shape[1]

    # ------------------------------------------------------------------
    # Baseline: c_i (scalar) → c_j  — cached, model-independent
    # ------------------------------------------------------------------
    if baseline_cache_path and os.path.exists(baseline_cache_path):
        print(f"  Loading baseline cache: {baseline_cache_path}")
        cached = joblib.load(baseline_cache_path)
        acc_baseline  = cached["acc_baseline"]
        loss_baseline = cached["loss_baseline"]
    else:
        print(f"  Computing baseline probes ({K}×{K}, n_jobs={n_jobs})...")
        baseline_X_tr = [c_true_train[:, i].reshape(-1, 1) for i in range(K)]
        baseline_X_te = [c_true_test[:, i].reshape(-1, 1)  for i in range(K)]

        rows = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
            delayed(_probe_row)(i, K, baseline_X_tr[i], baseline_X_te[i],
                                c_true_train, c_true_test)
            for i in range(K)
        )
        acc_baseline  = np.ones((K, K))
        loss_baseline = np.zeros((K, K))
        for i, acc_row, loss_row in rows:
            acc_baseline[i]  = acc_row
            loss_baseline[i] = loss_row

        if baseline_cache_path:
            os.makedirs(os.path.dirname(baseline_cache_path), exist_ok=True)
            joblib.dump({"acc_baseline": acc_baseline, "loss_baseline": loss_baseline},
                        baseline_cache_path)
            print(f"  Baseline cache saved: {baseline_cache_path}")

    # ------------------------------------------------------------------
    # Full probe: [ĉ_i, c_i] → c_j  — model-dependent, not cached
    # ------------------------------------------------------------------
    print(f"  Computing concat probes ({K}×{K}, n_jobs={n_jobs})...")
    concat_X_tr = [
        np.concatenate([c_mix_train[:, i, :], c_true_train[:, i].reshape(-1, 1)], axis=1)
        for i in range(K)
    ]
    concat_X_te = [
        np.concatenate([c_mix_test[:, i, :], c_true_test[:, i].reshape(-1, 1)], axis=1)
        for i in range(K)
    ]

    rows = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        delayed(_probe_row)(i, K, concat_X_tr[i], concat_X_te[i],
                            c_true_train, c_true_test)
        for i in range(K)
    )
    acc_concat  = np.ones((K, K))
    loss_concat = np.zeros((K, K))
    for i, acc_row, loss_row in rows:
        acc_concat[i]  = acc_row
        loss_concat[i] = loss_row

    # ------------------------------------------------------------------
    # Differences
    # ------------------------------------------------------------------
    acc_diff       = acc_concat - acc_baseline
    loss_diff      = loss_concat - loss_baseline
    loss_diff_norm = np.where(
        np.abs(loss_baseline) > 1e-10,
        loss_diff / loss_baseline,
        0.0,
    )
    # Diagonal: acc=1, all diffs=0 (trivially perfect on both probes)
    np.fill_diagonal(acc_diff,       0.0)
    np.fill_diagonal(loss_diff,      0.0)
    np.fill_diagonal(loss_diff_norm, 0.0)

    return {
        "acc_baseline":   acc_baseline,
        "acc_concat":     acc_concat,
        "acc_diff":       acc_diff,
        "loss_baseline":  loss_baseline,
        "loss_concat":    loss_concat,
        "loss_diff":      loss_diff,
        "loss_diff_norm": loss_diff_norm,
    }


def print_icl_probe_summary(icl, label=""):
    """Print mean off-diagonal stats for the ICL probe matrices."""
    K = icl["acc_diff"].shape[0]
    mask = ~np.eye(K, dtype=bool)
    print(f"\n  ICL probe summary {label}")
    print(f"    acc_baseline mean  : {icl['acc_baseline'][mask].mean():.4f}")
    print(f"    acc_concat   mean  : {icl['acc_concat'][mask].mean():.4f}")
    print(f"    acc_diff     mean  : {icl['acc_diff'][mask].mean():.4f}  "
          f"(+ve = embedding leaks info about other concepts)")
    print(f"    loss_diff    mean  : {icl['loss_diff'][mask].mean():.4f}  "
          f"(-ve = embedding reduces loss)")
    print(f"    loss_diff_norm mean: {icl['loss_diff_norm'][mask].mean():.4f}")
