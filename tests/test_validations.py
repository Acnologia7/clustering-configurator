import pytest
from pydantic import ValidationError

from app.utils.validations import (
    AgglomerativeConfig,
    ClusteringConfig,
    Config,
    DBSCANConfig,
    KMeansConfig,
)


def test_kmeans_config_default():
    """Test default values for KMeansConfig."""
    config = KMeansConfig()
    assert config.n_clusters == 8
    assert config.init == "k-means++"
    assert config.algorithm == "lloyd"


def test_kmeans_config_custom():
    """Test custom initialization of KMeansConfig."""
    config = KMeansConfig(n_clusters=5, init="random", max_iter=500)
    assert config.n_clusters == 5
    assert config.init == "random"
    assert config.max_iter == 500


def test_kmeans_config_invalid_init():
    """Test invalid value for KMeansConfig init."""
    with pytest.raises(ValidationError):
        KMeansConfig(init="invalid_method")


def test_dbscan_config_default():
    """Test default values for DBSCANConfig."""
    config = DBSCANConfig()
    assert config.eps == 0.5
    assert config.min_samples == 5
    assert config.algorithm == "auto"


def test_dbscan_config_custom():
    """Test custom initialization of DBSCANConfig."""
    config = DBSCANConfig(eps=0.3, min_samples=10, algorithm="kd_tree")
    assert config.eps == 0.3
    assert config.min_samples == 10
    assert config.algorithm == "kd_tree"


def test_agglomerative_config_default():
    """Test default values for AgglomerativeConfig."""
    config = AgglomerativeConfig()
    assert config.n_clusters == 2
    assert config.linkage == "ward"
    assert config.compute_distances is False


def test_agglomerative_config_custom():
    """Test custom initialization of AgglomerativeConfig."""
    config = AgglomerativeConfig(
        n_clusters=4, linkage="average", distance_threshold=None
    )
    assert config.n_clusters == 4
    assert config.linkage == "average"
    assert config.distance_threshold is None


def test_clustering_config_kmeans():
    """Test ClusteringConfig with KMeansConfig."""
    config = ClusteringConfig(
        input_format="json",
        output_format="numpy",
        algorithm="KMeans",
        hyperparameters={"n_clusters": 3},
    )
    assert config.algorithm == "KMeans"
    assert isinstance(config.hyperparameters, KMeansConfig)
    assert config.hyperparameters.n_clusters == 3


def test_clustering_config_dbscan():
    """Test ClusteringConfig with DBSCANConfig."""
    config = ClusteringConfig(
        input_format="numpy",
        output_format="json",
        algorithm="DBSCAN",
        hyperparameters={"eps": 0.2, "min_samples": 10},
    )
    assert config.algorithm == "DBSCAN"
    assert isinstance(config.hyperparameters, DBSCANConfig)
    assert config.hyperparameters.eps == 0.2
    assert config.hyperparameters.min_samples == 10


def test_clustering_config_invalid_algorithm():
    """Test ClusteringConfig with invalid algorithm."""
    with pytest.raises(ValidationError):
        ClusteringConfig(
            input_format="json",
            output_format="numpy",
            algorithm="InvalidAlgorithm",
            hyperparameters={},
        )


def test_clustering_config_agglomerative():
    """Test ClusteringConfig with AgglomerativeConfig."""
    config = ClusteringConfig(
        input_format="json",
        output_format="numpy",
        algorithm="AgglomerativeClustering",
        hyperparameters={"n_clusters": 3, "linkage": "average"},
    )
    assert config.algorithm == "AgglomerativeClustering"
    assert isinstance(config.hyperparameters, AgglomerativeConfig)
    assert config.hyperparameters.n_clusters == 3
    assert config.hyperparameters.linkage == "average"


def test_config_root():
    """Test root Config object."""
    config = Config(
        clustering={
            "input_format": "json",
            "output_format": "numpy",
            "algorithm": "KMeans",
            "hyperparameters": {"n_clusters": 4},
        }
    )
    assert config.clustering.algorithm == "KMeans"
    assert config.clustering.hyperparameters.n_clusters == 4
