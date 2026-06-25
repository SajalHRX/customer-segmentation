"""Unit tests for src.clustering (Phase-4 machinery, docs 10/11).

Strategy: build synthetic data whose answer we KNOW -- three well-separated Gaussian blobs
(true K = 3) -- and assert the machinery recovers it. Everything is seeded so it reproduces.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from src import clustering, utils


@pytest.fixture(scope="module")
def blobs():
    """150 points in 3 tight, well-separated blobs -> true K = 3 (a known-answer fixture)."""
    X, y = make_blobs(
        n_samples=150, centers=3, cluster_std=0.4, center_box=(-12, 12),
        random_state=utils.RANDOM_SEED,
    )
    df = pd.DataFrame(X, columns=["f0", "f1"], index=[f"c{i}" for i in range(len(X))])
    return df, y


# --------------------------------------------------------------------------- fitting + labels
def test_fitters_return_k_clusters_and_labels_align(blobs):
    df, _ = blobs
    for fit in (clustering.fit_kmeans, clustering.fit_gmm, clustering.fit_ward):
        model = fit(df, 3)
        labels = clustering.labels_of(model, df)
        assert labels.shape == (len(df),)
        assert len(np.unique(labels)) == 3

    # K-Means re-predicting the training matrix reproduces its own training labels.
    km = clustering.fit_kmeans(df, 3)
    assert np.array_equal(km.labels_, clustering.labels_of(km, df))


def test_labels_of_ward_needs_no_matrix(blobs):
    df, _ = blobs
    ward = clustering.fit_ward(df, 3)
    # Agglomerative stores labels_, so labels_of works without the matrix.
    assert clustering.labels_of(ward).shape == (len(df),)


# --------------------------------------------------------------------------- K-selection indices
def test_sweep_k_silhouette_peaks_at_true_k(blobs):
    df, _ = blobs
    swept = clustering.sweep_k(df, range(2, 7), method="kmeans")
    assert list(swept.columns) == ["inertia", "silhouette", "calinski_harabasz", "davies_bouldin"]
    # Three separated blobs -> silhouette and CH maximised, DB minimised at K = 3.
    assert swept["silhouette"].idxmax() == 3
    assert swept["calinski_harabasz"].idxmax() == 3
    assert swept["davies_bouldin"].idxmin() == 3
    # Inertia is monotonically non-increasing in K (the elbow input).
    assert swept["inertia"].is_monotonic_decreasing


def test_sweep_k_rejects_k_below_2(blobs):
    df, _ = blobs
    with pytest.raises(ValueError):
        clustering.sweep_k(df, [1, 2], method="kmeans")


def test_knee_point_finds_synthetic_elbow():
    # A convex-decreasing curve with an obvious bend at k = 3.
    ks = [1, 2, 3, 4, 5, 6]
    inertia = [100.0, 40.0, 20.0, 18.0, 16.5, 15.5]
    assert clustering.knee_point(ks, inertia) == 3


def test_gap_statistic_chooses_true_k(blobs):
    df, _ = blobs
    gap = clustering.gap_statistic(df, range(1, 6), n_refs=10, reference="uniform")
    assert list(gap.columns) == ["log_W", "gap", "s_k"]
    # The gap rule (and its peak) should land on the true K = 3 for clean blobs.
    assert clustering.gap_choice(gap) == 3


def test_gmm_bic_prefers_true_k(blobs):
    df, _ = blobs
    bic = clustering.gmm_bic(df, range(1, 6), covariance_types=("full",))
    assert bic.index.name == "k"
    assert bic["full"].idxmin() == 3   # BIC (lower better) bottoms out at the true K


# --------------------------------------------------------------------------- stability
def test_bootstrap_stability_high_on_clean_blobs(blobs):
    df, _ = blobs
    jac = clustering.bootstrap_stability(df, 3, method="kmeans", n_boot=30, frac=0.8)
    assert len(jac) == 3
    # Well-separated blobs reproduce under resampling -> every cluster clears the doc's
    # "stable" threshold (mean Jaccard > 0.75).
    assert jac.min() > 0.75


def test_consensus_matrix_blocks(blobs):
    df, y = blobs
    cons = clustering.consensus_matrix(df, 3, method="kmeans", n_runs=20, frac=0.8)
    assert cons.shape == (len(df), len(df))
    assert np.allclose(np.diag(cons), 1.0)
    assert cons.min() >= 0.0 and cons.max() <= 1.0
    # Two points in the SAME true blob co-cluster often; two in DIFFERENT blobs rarely.
    same = np.flatnonzero(y == y[0])
    diff = np.flatnonzero(y != y[0])
    assert cons[same[0], same[1]] > 0.8
    assert cons[diff[0], diff[-1]] < 0.2


# --------------------------------------------------------------------------- comparison
def test_ari_matrix_identity_and_agreement(blobs):
    df, _ = blobs
    km = clustering.labels_of(clustering.fit_kmeans(df, 3), df)
    ward = clustering.labels_of(clustering.fit_ward(df, 3))
    mat = clustering.ari_matrix({"kmeans": km, "ward": ward})
    # Self-agreement is exactly 1; the two methods agree strongly on clean blobs.
    assert mat.loc["kmeans", "kmeans"] == pytest.approx(1.0)
    assert mat.loc["ward", "ward"] == pytest.approx(1.0)
    assert mat.loc["kmeans", "ward"] > 0.9
    # ARI is symmetric.
    assert mat.loc["kmeans", "ward"] == pytest.approx(mat.loc["ward", "kmeans"])


def test_cophenetic_corr_high_on_clean_blobs(blobs):
    df, _ = blobs
    coph = clustering.cophenetic_corr(df)
    assert -1.0 <= coph <= 1.0
    assert coph > 0.7   # the dendrogram faithfully represents clean, well-separated data


def test_pca_2d_shape_and_index(blobs):
    df, _ = blobs
    coords = clustering.pca_2d(df)
    assert list(coords.columns) == ["PC1", "PC2"]
    assert list(coords.index) == list(df.index)        # index preserved
    assert len(coords.attrs["explained"]) == 2         # one ratio per component
