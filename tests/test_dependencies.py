import pytest

from app.clustering import CDBSCAN, CAgglomerativeClustering, CKMeans
from app.dependencies import ClusteringContainer
from app.utils.io_handler import JSONHandler, NumpyHandler


@pytest.fixture
def container_with_kmeans():
    """Fixture for ClusteringContainer configured to use KMeans."""
    container = ClusteringContainer()
    container.config.from_dict(
        {
            "clustering": {
                "input_format": "json",
                "output_format": "numpy",
                "algorithm": "KMeans",
                "hyperparameters": {
                    "n_clusters": 3,
                    "init": "k-means++",
                    "n_init": 10,
                    "max_iter": 300,
                    "tol": 1e-4,
                    "verbose": 0,
                    "random_state": 42,
                    "copy_x": True,
                    "algorithm": "auto",
                },
            }
        }
    )
    return container


@pytest.fixture
def container_with_dbscan():
    """Fixture for ClusteringContainer configured to use DBSCAN."""
    container = ClusteringContainer()
    container.config.from_dict(
        {
            "clustering": {
                "input_format": "numpy",
                "output_format": "numpy",
                "algorithm": "DBSCAN",
                "hyperparameters": {
                    "eps": 0.5,
                    "min_samples": 5,
                    "metric": "euclidean",
                    "metric_params": None,
                    "algorithm": "auto",
                    "leaf_size": 30,
                    "p": None,
                    "n_jobs": -1,
                },
            }
        }
    )
    return container


@pytest.fixture
def container_with_agglomerative():
    """
    Fixture for ClusteringContainer
    configured to use AgglomerativeClustering.
    """
    container = ClusteringContainer()
    container.config.from_dict(
        {
            "clustering": {
                "input_format": "json",
                "output_format": "numpy",
                "algorithm": "AgglomerativeClustering",
                "hyperparameters": {
                    "n_clusters": 3,
                    "metric": "euclidean",
                    "memory": None,
                    "connectivity": None,
                    "compute_full_tree": False,
                    "linkage": "ward",
                    "distance_threshold": None,
                    "compute_distances": "auto",
                },
            }
        }
    )
    return container


def test_kmeans_algorithm(container_with_kmeans):
    """
    Test that KMeans algorithm is
    instantiated with correct hyperparameters.
    """
    container = container_with_kmeans
    clustering_algorithm = container.clustering_algorithm()

    assert isinstance(clustering_algorithm, CKMeans)
    assert clustering_algorithm.model.n_clusters == 3
    assert clustering_algorithm.model.init == "k-means++"
    assert clustering_algorithm.model.n_init == 10
    assert clustering_algorithm.model.max_iter == 300
    assert clustering_algorithm.model.tol == 1e-4
    assert clustering_algorithm.model.verbose == 0
    assert clustering_algorithm.model.random_state == 42


def test_dbscan_algorithm(container_with_dbscan):
    """
    Test that DBSCAN algorithm is
    instantiated with correct hyperparameters.
    """
    container = container_with_dbscan
    clustering_algorithm = container.clustering_algorithm()

    assert isinstance(clustering_algorithm, CDBSCAN)
    assert clustering_algorithm.model.eps == 0.5
    assert clustering_algorithm.model.min_samples == 5
    assert clustering_algorithm.model.metric == "euclidean"
    assert clustering_algorithm.model.leaf_size == 30
    assert clustering_algorithm.model.n_jobs == -1


def test_agglomerative_algorithm(container_with_agglomerative):
    """
    Test that AgglomerativeClustering algorithm is
    instantiated with correct hyperparameters.
    """
    container = container_with_agglomerative
    clustering_algorithm = container.clustering_algorithm()

    assert isinstance(clustering_algorithm, CAgglomerativeClustering)
    assert clustering_algorithm.model.n_clusters == 3
    assert clustering_algorithm.model.metric == "euclidean"
    assert clustering_algorithm.model.linkage == "ward"


def test_input_handler_json(container_with_kmeans):
    """Test that the correct input handler is selected based on config."""
    container = container_with_kmeans
    input_handler = container.input_handler()

    assert isinstance(input_handler, JSONHandler)


def test_output_handler_numpy(container_with_dbscan):
    """Test that the correct output handler is selected based on config."""
    container = container_with_dbscan
    output_handler = container.output_handler()

    assert isinstance(output_handler, NumpyHandler)
