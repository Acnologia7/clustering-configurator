from unittest.mock import MagicMock, patch

import pytest
from numpy import array

from app.clustering import CDBSCAN, CAgglomerativeClustering, CKMeans


@pytest.fixture
def mock_data():
    """Fixture to provide sample input data."""
    return array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])


@patch("app.clustering.KMeans")
def test_ckmeans(mock_kmeans, mock_data):
    """Test CKMeans clustering."""
    mock_instance = MagicMock()
    mock_instance.fit_predict.return_value = [0, 0, 0, 1, 1, 1]
    mock_kmeans.return_value = mock_instance

    kmeans = CKMeans(
        n_clusters=2,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=42,
        copy_x=True,
        algorithm="lloyd",
    )
    labels = kmeans.cluster(mock_data)

    # Verify that the underlying KMeans
    # was initialized with the correct parameters
    mock_kmeans.assert_called_once_with(
        n_clusters=2,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=42,
        copy_x=True,
        algorithm="lloyd",
    )

    # Verify that fit_predict was called on the mock instance
    mock_instance.fit_predict.assert_called_once_with(
        mock_data, sample_weight=None
    )

    # Verify the output
    assert labels == [0, 0, 0, 1, 1, 1]


@patch("app.clustering.DBSCAN")
def test_cdbscan(mock_dbscan, mock_data):
    """Test CDBSCAN clustering."""
    mock_instance = MagicMock()
    mock_instance.fit_predict.return_value = [0, 0, 0, -1, -1, -1]
    mock_dbscan.return_value = mock_instance

    dbscan = CDBSCAN(
        eps=3.0,
        min_samples=2,
        metric="euclidean",
        metric_params=None,
        algorithm="auto",
        leaf_size=30,
        p=None,
        n_jobs=-1,
    )
    labels = dbscan.cluster(mock_data)

    # Verify that the underlying DBSCAN
    # was initialized with the correct parameters
    mock_dbscan.assert_called_once_with(
        eps=3.0,
        min_samples=2,
        metric="euclidean",
        metric_params=None,
        algorithm="auto",
        leaf_size=30,
        p=None,
        n_jobs=-1,
    )

    # Verify that fit_predict was called on the mock instance
    mock_instance.fit_predict.assert_called_once_with(
        mock_data, sample_weight=None
    )

    # Verify the output
    assert labels == [0, 0, 0, -1, -1, -1]


@patch("app.clustering.AgglomerativeClustering")
def test_cagglomerative_clustering(mock_agglomerative, mock_data):
    """Test CAgglomerativeClustering clustering."""
    mock_instance = MagicMock()
    mock_instance.fit_predict.return_value = [0, 0, 0, 1, 0, 1]
    mock_agglomerative.return_value = mock_instance

    agglomerative = CAgglomerativeClustering(
        n_clusters=2,
        metric="euclidean",
        memory=None,
        connectivity=None,
        compute_full_tree="auto",
        linkage="ward",
        distance_threshold=None,
        compute_distances=False,
    )
    labels = agglomerative.cluster(mock_data)

    # Verify that the underlying AgglomerativeClustering
    # was initialized with the correct parameters
    mock_agglomerative.assert_called_once_with(
        n_clusters=2,
        metric="euclidean",
        memory=None,
        connectivity=None,
        compute_full_tree="auto",
        linkage="ward",
        distance_threshold=None,
        compute_distances=False,
    )

    # Verify that fit_predict was called on the mock instance
    mock_instance.fit_predict.assert_called_once_with(mock_data)

    # Verify the output
    assert labels == [0, 0, 0, 1, 0, 1]


@patch("app.clustering.AgglomerativeClustering")
def test_cagglomerative_clustering_with_sample_weights_exception(
    mock_agglomerative, mock_data
):
    """
    Test CAgglomerativeClustering
    does not call fit_predict with sample weights.
    """
    mock_instance = MagicMock()
    mock_agglomerative.return_value = mock_instance

    agglomerative = CAgglomerativeClustering(
        n_clusters=2,
        metric="euclidean",
        memory=None,
        connectivity=None,
        compute_full_tree="auto",
        linkage="ward",
        distance_threshold=None,
        compute_distances=False,
    )

    # Attempt to cluster with sample weights, expecting an exception
    with pytest.raises(
        NotImplementedError,
        match="CAgglomerativeClustering does not support `sample_weight`.",
    ):
        agglomerative.cluster(
            mock_data, sample_weight=array([1, 1, 1, 1, 1, 1])
        )

    # Ensure that fit_predict was never called
    mock_instance.fit_predict.assert_not_called()
