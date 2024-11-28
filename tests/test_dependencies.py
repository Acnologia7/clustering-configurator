import pytest
from numpy import array

from app.dependencies import ClusteringContainer


class MockKMeans:
    def cluster(self, data):
        # The mock cluster method just returns a list of 0s,
        # mimicking clustering behavior
        return [0] * len(data)


class MockDBSCAN:
    def cluster(self, data):
        # The mock cluster method just returns a list of 1s,
        # mimicking clustering behavior
        return [1] * len(data)


class MockAgglomerativeClustering:
    def cluster(self, data):
        # The mock cluster method just returns a list of 2s,
        #  mimicking clustering behavior
        return [2] * len(data)


class MockJSONHandler:
    def save(self, data, path):
        # Simulate saving to JSON format
        return f"Saving {len(data)} items to JSON at {path}"


class MockNumpyHandler:
    def save(self, data, path):
        # Simulate saving to NumPy format
        return f"Saving {len(data)} items to NumPy at {path}"


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

    # Manually overriding the providers with mock classes
    container.clustering_algorithm = MockKMeans
    container.input_handler = MockJSONHandler
    container.output_handler = MockNumpyHandler

    return container


@pytest.fixture
def container_with_dbscan():
    """Fixture for ClusteringContainer configured to use DBSCAN."""
    container = ClusteringContainer()

    container.config.from_dict(
        {
            "clustering": {
                "input_format": "numpy",
                "output_format": "json",
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

    # Manually overriding the providers with mock classes
    container.clustering_algorithm = MockDBSCAN
    container.input_handler = MockNumpyHandler
    container.output_handler = MockJSONHandler

    return container


@pytest.fixture
def container_with_dbscan_numpy_output():
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

    # Manually overriding the providers with mock classes
    container.clustering_algorithm = MockDBSCAN
    container.input_handler = MockNumpyHandler
    container.output_handler = MockNumpyHandler

    return container


@pytest.fixture
def container_with_agglomerative():
    """
    Fixture for ClusteringContainer configured to
    use AgglomerativeClustering.
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

    # Manually overriding the providers with mock classes
    container.clustering_algorithm = MockAgglomerativeClustering
    container.input_handler = MockJSONHandler
    container.output_handler = MockNumpyHandler

    return container


def test_kmeans_algorithm(container_with_kmeans):
    """
    Test that the KMeans algorithm is selected correctly
    and functions as expected.
    """
    container = container_with_kmeans
    clustering_algorithm = (
        container.clustering_algorithm()
    )  # Should now return the MockKMeans instance
    data = array([[1, 2], [3, 4], [5, 6]])
    labels = clustering_algorithm.cluster(
        data
    )  # Calling cluster() on MockKMeans
    assert labels == [0, 0, 0], "Expected all labels to be 0 in KMeans mock"


def test_dbscan_algorithm(container_with_dbscan):
    """
    Test that the DBSCAN algorithm is
    selected correctly and functions as expected.
    """
    container = container_with_dbscan
    clustering_algorithm = (
        container.clustering_algorithm()
    )  # Should now return the MockDBSCAN instance
    data = array([[1, 2], [3, 4], [5, 6]])
    labels = clustering_algorithm.cluster(
        data
    )  # Calling cluster() on MockDBSCAN
    assert labels == [1, 1, 1], "Expected all labels to be 1 in DBSCAN mock"


def test_agglomerative_algorithm(container_with_agglomerative):
    """
    Test that the AgglomerativeClustering algorithm
    is selected correctly and functions as expected.
    """
    container = container_with_agglomerative
    clustering_algorithm = (
        container.clustering_algorithm()
    )  # Should now return the MockAgglomerativeClustering instance
    data = array([[1, 2], [3, 4], [5, 6]])
    labels = clustering_algorithm.cluster(
        data
    )  # Calling cluster() on MockAgglomerativeClustering
    assert labels == [
        2,
        2,
        2,
    ], "Expected all labels to be 2 in AgglomerativeClustering mock"


def test_io_handler_save_json(container_with_kmeans):
    """Test that the IOHandler saves data correctly to a JSON format."""
    container = container_with_kmeans
    io_handler = (
        container.input_handler()
    )  # Should return MockJSONHandler instance
    data = array([[1, 2], [3, 4], [5, 6]])
    result = io_handler.save(data, "output.json")
    assert (
        result == "Saving 3 items to JSON at output.json"
    ), "Expected JSON save behavior"


def test_io_handler_save_numpy(container_with_dbscan_numpy_output):
    """Test that the IOHandler saves data correctly to a NumPy format."""
    container = container_with_dbscan_numpy_output

    io_handler = (
        container.output_handler()
    )  # Should return MockNumpyHandler instance
    data = array([[1, 2], [3, 4], [5, 6]])
    result = io_handler.save(data, "output.npy")
    assert (
        result == "Saving 3 items to NumPy at output.npy"
    ), "Expected NumPy save behavior"
