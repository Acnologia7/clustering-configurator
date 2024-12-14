from dependency_injector import containers, providers

from app.clustering import CDBSCAN, CAgglomerativeClustering, CKMeans
from app.utils.io_handler import JSONHandler, NumpyHandler


class ClusteringContainer(containers.DeclarativeContainer):
    """
    Dependency Injection container for
    clustering algorithms and I/O operations.

    Initializes clustering algorithms
    (CKMeans, CDBSCAN, AgglomerativeClustering)
    and I/O handlers
    (JSONHandler, NumpyHandler) based on configuration.

    Components:
    - **config**:
    Stores clustering and handler configurations.
    - **input_handler**:
    Selects input handler based on `config.clustering.input_format`.
    - **output_handler**:
    Selects output handler based on `config.clustering.output_format`.
    - **clustering_algorithm**:
    Initializes the selected algorithm from `config.clustering.algorithm`.

    Example:
    ```python
    config = {
        "clustering": {
            "input_format": "json",
            "output_format": "numpy",
            "algorithm": "DBSCAN",
            "hyperparameters": {
                "eps": 0.5,
                "min_samples": 5,
                "metric": "euclidean",
                "leaf_size": 30,
                "n_jobs": -1,
            }
        }
    }

    container = ClusteringContainer()
    container.config.from_dict(config)
    clustering = container.clustering_algorithm()
    io_handler = container.io_handler()

    data = array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    labels = clustering.cluster(data)
    print("Cluster Labels:", labels)
    io_handler.save_data(data, "data_output_path")
    """

    config = providers.Configuration()

    input_handler = providers.Selector(
        config.clustering.input_format,
        json=providers.Factory(JSONHandler),
        numpy=providers.Factory(NumpyHandler),
    )

    output_handler = providers.Selector(
        config.clustering.output_format,
        json=providers.Factory(JSONHandler),
        numpy=providers.Factory(NumpyHandler),
    )

    clustering_algorithm = providers.Selector(
        config.clustering.algorithm,
        KMeans=providers.Factory(
            CKMeans,
            n_clusters=config.clustering.hyperparameters.n_clusters,
            init=config.clustering.hyperparameters.init,
            n_init=config.clustering.hyperparameters.n_init,
            max_iter=config.clustering.hyperparameters.max_iter,
            tol=config.clustering.hyperparameters.tol,
            verbose=config.clustering.hyperparameters.verbose,
            random_state=config.clustering.hyperparameters.random_state,
            copy_x=config.clustering.hyperparameters.copy_x,
            algorithm=config.clustering.hyperparameters.algorithm,
        ),
        DBSCAN=providers.Factory(
            CDBSCAN,
            eps=config.clustering.hyperparameters.eps,
            min_samples=config.clustering.hyperparameters.min_samples,
            metric=config.clustering.hyperparameters.metric,
            metric_params=config.clustering.hyperparameters.metric_params,
            algorithm=config.clustering.hyperparameters.algorithm,
            leaf_size=config.clustering.hyperparameters.leaf_size,
            p=config.clustering.hyperparameters.p,
            n_jobs=config.clustering.hyperparameters.n_jobs,
        ),
        AgglomerativeClustering=providers.Factory(
            CAgglomerativeClustering,
            n_clusters=config.clustering.hyperparameters.n_clusters,
            metric=config.clustering.hyperparameters.metric,
            memory=config.clustering.hyperparameters.memory,
            connectivity=config.clustering.hyperparameters.connectivity,
            compute_full_tree=(
                config.clustering.hyperparameters.compute_full_tree
            ),
            linkage=config.clustering.hyperparameters.linkage,
            distance_threshold=(
                config.clustering.hyperparameters.distance_threshold
            ),
            compute_distances=(
                config.clustering.hyperparameters.compute_distances
            ),
        ),
    )
