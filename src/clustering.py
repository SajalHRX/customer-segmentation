"""Clustering machinery: the SHARED methods both Phase-4 notebooks import (docs 10, 11).

All reusable logic lives here once (unit-tested); the notebooks (`04a_choosing_k`,
`04b_method_comparison`) only *call* these functions. Nothing here decides K or names a
segment -- that is the notebooks' job. The two design docs:

- doc 10 (Choosing & Validating K): ``sweep_k``, ``knee_point``, ``gap_statistic``,
  ``gmm_bic``, ``bootstrap_stability``, ``consensus_matrix``.
- doc 11 (Method Comparison):       ``fit_kmeans`` / ``fit_gmm`` / ``fit_ward``, ``labels_of``,
  ``ari_matrix``, ``cophenetic_corr``, ``pca_2d``.

Everything is seeded with ``utils.RANDOM_SEED`` so results reproduce. Inputs are the scaled
clustering matrix (a DataFrame or ndarray of shape n_customers x n_features); functions accept
either via the small ``_as_array`` shim and never mutate their input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import utils


def _as_array(matrix) -> np.ndarray:
    """Accept a DataFrame or ndarray and return a float ndarray (the algorithms want raw values)."""
    if isinstance(matrix, pd.DataFrame):
        return matrix.to_numpy(dtype=float)
    return np.asarray(matrix, dtype=float)


# --------------------------------------------------------------------------- fitting (doc 11)
def fit_kmeans(matrix, k: int, seed: int = utils.RANDOM_SEED, n_init: int = 10):
    """Fit K-Means with k clusters -- the deployed baseline / quantizer (doc 11).

    ``n_init`` random restarts guard against bad initialisations (the cheap seed-stability the
    doc calls for); the best (lowest-inertia) run is kept. Returns the fitted estimator so the
    caller can read ``.labels_``, ``.cluster_centers_`` (interpretable archetypes) and ``.inertia_``.
    """
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(_as_array(matrix))


def fit_gmm(matrix, k: int, seed: int = utils.RANDOM_SEED,
            covariance_type: str = "full", n_init: int = 5):
    """Fit a Gaussian Mixture with k components -- the continuum-aware cross-check (doc 11).

    GMM generalises K-Means: each component has its own mean (archetype) and covariance (the
    oval's shape/size/tilt), and membership is SOFT (``predict_proba``) -- so it can flag
    fence-sitters and vote on K via ``.bic``. ``covariance_type`` is the bias-variance dial
    (spherical ~ soft K-Means, full = most flexible); chosen by BIC, not by eye (doc 11).
    Fitted by EM to a LOCAL optimum, so ``n_init`` restarts matter.
    """
    from sklearn.mixture import GaussianMixture

    return GaussianMixture(
        n_components=k, covariance_type=covariance_type, n_init=n_init, random_state=seed
    ).fit(_as_array(matrix))


def fit_ward(matrix, k: int):
    """Fit Ward agglomerative clustering with k clusters -- structural corroboration (doc 11).

    Ward merges the pair that adds the least within-cluster variance (the K-Means-like linkage),
    giving a dendrogram you can read as the continuum's family tree. Deterministic (no seed).
    Returns the fitted estimator; read ``.labels_`` (it has no out-of-sample ``predict``).
    """
    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit(_as_array(matrix))


def labels_of(model, matrix=None) -> np.ndarray:
    """Return integer cluster labels from any of the three fitted models.

    K-Means/GMM expose ``predict`` (and re-predicting on the training matrix reproduces the
    training labels); Agglomerative only stores ``labels_``. This shim hides that difference so
    downstream code (ARI, profiling) treats all three uniformly.
    """
    if hasattr(model, "labels_"):
        return np.asarray(model.labels_, dtype=int)
    if matrix is None:
        raise ValueError("this model has no .labels_; pass the matrix so predict() can run")
    return np.asarray(model.predict(_as_array(matrix)), dtype=int)


# --------------------------------------------------------------------------- K-selection indices (doc 10)
def sweep_k(matrix, k_range, method: str = "kmeans", seed: int = utils.RANDOM_SEED) -> pd.DataFrame:
    """Score a range of K on the internal validity indices -- the K-selection panel (doc 10).

    For each K in ``k_range`` fit ``method`` (kmeans|gmm|ward) and compute, on one row:

    - ``inertia``           = within-cluster sum of squares (only for K-Means; the elbow input)
    - ``silhouette``        = mean (b - a)/max(a,b); HIGHER better (the workhorse)
    - ``calinski_harabasz`` = between/within variance ratio; HIGHER better
    - ``davies_bouldin``    = avg worst-case cluster-pair similarity; LOWER better

    Silhouette/CH/DB all implicitly favour convex, equal-size clusters -- so they corroborate,
    they do not crown a winner alone (doc 10). K must be >= 2 for the indices to be defined.
    """
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    fitters = {"kmeans": fit_kmeans, "gmm": fit_gmm, "ward": fit_ward}
    if method not in fitters:
        raise ValueError(f"method must be one of {list(fitters)}, got {method!r}")

    X = _as_array(matrix)
    rows = []
    for k in k_range:
        if k < 2:
            raise ValueError("silhouette/CH/DB need k >= 2")
        model = fitters[method](X, k) if method == "ward" else fitters[method](X, k, seed=seed)
        labels = labels_of(model, X)
        rows.append({
            "k": k,
            "inertia": float(getattr(model, "inertia_", np.nan)),
            "silhouette": float(silhouette_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
        })
    return pd.DataFrame(rows).set_index("k")


def knee_point(k_values, curve) -> int:
    """Objective elbow of a decreasing curve -- a dependency-free Kneedle (doc 10).

    The elbow is subjective by eye, so we make it reproducible: normalise both axes to [0, 1],
    draw the chord from the first point to the last, and return the K whose curve point sits
    FARTHEST BELOW that chord (the point of maximum curvature). This is the standard geometric
    knee and the clustering analogue of a PCA scree elbow. Use for intuition / to NARROW the
    range, never as the sole justification for K (doc 10).
    """
    k = np.asarray(list(k_values), dtype=float)
    y = np.asarray(list(curve), dtype=float)
    if len(k) < 3:
        raise ValueError("need at least 3 points to locate a knee")

    # Normalise to a unit square so the chord geometry is scale-free.
    kn = (k - k.min()) / (k.max() - k.min())
    yn = (y - y.min()) / (y.max() - y.min())
    # Straight chord through the two endpoints, evaluated at each kn.
    chord = yn[0] + (yn[-1] - yn[0]) * kn
    # The knee is where the curve sits FARTHEST below that chord (max curvature). For a convex,
    # decreasing inertia curve this is the elbow.
    return int(k[int(np.argmax(chord - yn))])


def gap_statistic(matrix, k_range, n_refs: int = 10, reference: str = "pca",
                  seed: int = utils.RANDOM_SEED) -> pd.DataFrame:
    """Tibshirani gap statistic -- the principled K choice that can return K=1 (doc 10).

    Compares the clustering's tightness to clustering pure NULL data with the same spread:
    ``Gap(k) = mean_b log(W_kb*) - log(W_k)``, where ``W_k`` = K-Means inertia. It corrects for
    the fact that inertia always falls, so unlike the elbow it has a real optimum.

    ``reference`` builds the null box either ``"uniform"`` (a plain bounding box) or ``"pca"``
    (sample uniformly in the PCA-rotated box, then rotate back -- the doc-preferred, more
    conservative null that respects the data's shape). Returns per-K ``log_W``, ``gap``, and the
    reference standard error ``s_k``; the caller applies the selection rule (smallest K with
    ``gap(k) >= gap(k+1) - s(k+1)``) via :func:`gap_choice`.
    """
    rng = np.random.default_rng(seed)
    X = _as_array(matrix)

    def _log_w(data, k):
        # Pooled within-cluster sum of squares = K-Means inertia (Tibshirani's W_k).
        return np.log(fit_kmeans(data, k, seed=seed).inertia_)

    # Pre-compute the bounding box (uniform) or the PCA rotation (pca) for the null draws.
    if reference == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(random_state=seed).fit(X)
        Xr = pca.transform(X)
        lo, hi = Xr.min(axis=0), Xr.max(axis=0)
    elif reference == "uniform":
        lo, hi = X.min(axis=0), X.max(axis=0)
    else:
        raise ValueError("reference must be 'uniform' or 'pca'")

    rows = []
    for k in k_range:
        log_w = _log_w(X, k)
        ref_logs = np.empty(n_refs)
        for b in range(n_refs):
            sample = rng.uniform(lo, hi, size=X.shape)
            if reference == "pca":
                sample = pca.inverse_transform(sample)  # rotate the null box back to data space
            ref_logs[b] = _log_w(sample, k)
        gap = ref_logs.mean() - log_w
        s_k = ref_logs.std(ddof=0) * np.sqrt(1.0 + 1.0 / n_refs)
        rows.append({"k": k, "log_W": log_w, "gap": gap, "s_k": s_k})
    return pd.DataFrame(rows).set_index("k")


def gap_choice(gap_df: pd.DataFrame) -> int:
    """Apply Tibshirani's selection rule to a :func:`gap_statistic` table.

    Smallest K such that ``gap(k) >= gap(k+1) - s(k+1)`` -- a built-in Occam's razor (the first K
    that is 'good enough' once the next K's noise is accounted for). Falls back to the max-gap K
    if no K satisfies the rule (e.g. monotone gap over the searched range).
    """
    ks = list(gap_df.index)
    for i in range(len(ks) - 1):
        k, k_next = ks[i], ks[i + 1]
        if gap_df.loc[k, "gap"] >= gap_df.loc[k_next, "gap"] - gap_df.loc[k_next, "s_k"]:
            return int(k)
    return int(gap_df["gap"].idxmax())


def gmm_bic(matrix, k_range, covariance_types=("spherical", "diag", "tied", "full"),
            seed: int = utils.RANDOM_SEED) -> pd.DataFrame:
    """BIC of a GMM across K x covariance_type -- a K vote from a DIFFERENT evidence family (doc 11).

    ``BIC = -2 ln L + p ln n`` (LOWER better); its ``ln n`` penalty is harsher than AIC's, so it
    guards against GMM adding spurious components just to patch non-Gaussian shape. Returns a
    (K x covariance_type) frame of BIC values; the best cell (min BIC) is the GMM's preferred
    (K, covariance) pair -- read alongside silhouette/gap, not instead of them.
    """
    X = _as_array(matrix)
    out = {}
    for cov in covariance_types:
        out[cov] = {int(k): float(fit_gmm(X, k, seed=seed, covariance_type=cov).bic(X))
                    for k in k_range}
    bic = pd.DataFrame(out)
    bic.index.name = "k"
    return bic


# --------------------------------------------------------------------------- stability (doc 10)
def bootstrap_stability(matrix, k: int, method: str = "kmeans", n_boot: int = 100,
                        frac: float = 0.8, seed: int = utils.RANDOM_SEED) -> pd.Series:
    """Per-cluster bootstrap Jaccard stability -- proving the segments are REAL (doc 10).

    Real structure reproduces under perturbation; artifacts do not. We cluster the full data once
    (the reference), then ``n_boot`` times SUBSAMPLE ``frac`` of customers WITHOUT replacement
    (replacement creates distance-0 duplicates -> degenerate micro-clusters), re-cluster, and for
    each reference cluster record its best Jaccard overlap (|A & B| / |A | B|) with a bootstrap
    cluster -- matched on the customers common to both runs. The mean over runs scores EACH
    cluster: >0.85 highly stable, 0.75-0.85 stable, 0.60-0.75 suggestive, <0.60 dissolves (often
    a sign K is one too high). Returns a Series indexed by reference cluster id.
    """
    rng = np.random.default_rng(seed)
    X = _as_array(matrix)
    n = X.shape[0]

    fitters = {"kmeans": fit_kmeans, "gmm": fit_gmm, "ward": fit_ward}
    fit = (lambda data: fitters[method](data, k)) if method == "ward" \
        else (lambda data: fitters[method](data, k, seed=seed))

    ref = labels_of(fit(X), X)
    ref_clusters = {c: set(np.flatnonzero(ref == c)) for c in np.unique(ref)}
    best = {c: [] for c in ref_clusters}

    size = max(k, int(round(frac * n)))  # never subsample below k points
    for _ in range(n_boot):
        idx = rng.choice(n, size=size, replace=False)
        boot = labels_of(fit(X[idx]), X[idx])
        # Map bootstrap rows back to ORIGINAL indices so overlaps are comparable.
        boot_clusters = [set(idx[boot == c]) for c in np.unique(boot)]
        for c, members in ref_clusters.items():
            best[c].append(max((_jaccard(members, b) for b in boot_clusters), default=0.0))

    return pd.Series({c: float(np.mean(v)) for c, v in best.items()}, name="mean_jaccard")


def _jaccard(a: set, b: set) -> float:
    """Jaccard overlap |a & b| / |a | b| on the customers the two runs share."""
    common = a | b
    return len(a & b) / len(common) if common else 0.0


def consensus_matrix(matrix, k: int, method: str = "kmeans", n_runs: int = 50,
                     frac: float = 0.8, seed: int = utils.RANDOM_SEED) -> np.ndarray:
    """Monti consensus matrix -- an independent SECOND way to judge K (doc 10).

    Entry (i, j) = fraction of runs (each on a ``frac`` subsample) in which customers i and j
    BOTH appear AND land in the same cluster. For a good K the (reordered) matrix is near
    block-diagonal with entries near 0 or 1; mushy K gives grey middling values. Returns the
    n x n consensus array (caller reorders/plots it as a heatmap, or scores its crispness).

    Memory note: n x n floats -- fine for a few thousand customers; subsample the matrix upstream
    if it bites (doc 10).
    """
    rng = np.random.default_rng(seed)
    X = _as_array(matrix)
    n = X.shape[0]

    fitters = {"kmeans": fit_kmeans, "gmm": fit_gmm, "ward": fit_ward}
    fit = (lambda data: fitters[method](data, k)) if method == "ward" \
        else (lambda data: fitters[method](data, k, seed=seed))

    co_assign = np.zeros((n, n))   # times pair i,j put in the SAME cluster
    co_sample = np.zeros((n, n))   # times pair i,j were BOTH sampled (the denominator)
    size = max(k, int(round(frac * n)))
    for _ in range(n_runs):
        idx = np.sort(rng.choice(n, size=size, replace=False))
        labels = labels_of(fit(X[idx]), X[idx])
        co_sample[np.ix_(idx, idx)] += 1
        for c in np.unique(labels):
            members = idx[labels == c]
            co_assign[np.ix_(members, members)] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        consensus = np.where(co_sample > 0, co_assign / co_sample, 0.0)
    np.fill_diagonal(consensus, 1.0)
    return consensus


# --------------------------------------------------------------------------- comparison (doc 11)
def ari_matrix(label_sets: dict) -> pd.DataFrame:
    """Pairwise Adjusted Rand Index among methods -- the headline credibility result (doc 11).

    ARI counts customer PAIRS two labelings agree on (together-together or apart-apart),
    chance-corrected to 0 = random, 1 = identical, and is label-invariant (immune to the
    'cluster 2 is arbitrary' problem). ``label_sets`` maps method name -> label array (all the
    SAME length / same customers). A high off-diagonal (>= 0.7) means three differently-biased
    methods agree on who-belongs-with-whom -> the segmentation is REAL, not one method's artifact.
    """
    from sklearn.metrics import adjusted_rand_score

    names = list(label_sets)
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            mat.loc[a, b] = adjusted_rand_score(label_sets[a], label_sets[b])
    return mat


def cophenetic_corr(matrix) -> float:
    """Cophenetic correlation of the Ward dendrogram -- is the tree honest? (doc 11).

    corr(original pairwise distances, tree-implied 'merge-height' distances). > 0.7 means the
    dendrogram faithfully represents the data's distance structure (so reading a cut off it is
    trustworthy). One number that validates the hierarchical view.
    """
    from scipy.cluster.hierarchy import cophenet, linkage
    from scipy.spatial.distance import pdist

    X = _as_array(matrix)
    dists = pdist(X)
    Z = linkage(X, method="ward")
    coph, _ = cophenet(Z, dists)
    return float(coph)


def pca_2d(matrix, seed: int = utils.RANDOM_SEED) -> pd.DataFrame:
    """Project to 2 principal components for label-overlay scatter plots (doc 11 observability).

    A shared 2-D view so every method's labels are plotted on the SAME coordinates (only the
    colours differ). Returns a DataFrame with ``PC1``/``PC2`` (index preserved if a DataFrame was
    passed) and the per-component explained-variance ratio stored on ``.attrs['explained']``.
    """
    from sklearn.decomposition import PCA

    X = _as_array(matrix)
    pca = PCA(n_components=2, random_state=seed)
    coords = pca.fit_transform(X)
    index = matrix.index if isinstance(matrix, pd.DataFrame) else None
    out = pd.DataFrame(coords, columns=["PC1", "PC2"], index=index)
    out.attrs["explained"] = pca.explained_variance_ratio_
    return out
