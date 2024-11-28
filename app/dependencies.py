from dependency_injector import containers, providers

from app.clustering import CDBSCAN, CAgglomerativeClustering, CKMeans
from app.utils.io_handler import IOHandler, JSONHandler, NumpyHandler


class ClusteringContainer(containers.DeclarativeContainer):
    """
    Dependency Injection container for clustering and I/O operations.

    This container centralizes configuration and
    initialization of various clustering
    algorithms and input/output handlers.
    It leverages the `dependency_injector` library
    to dynamically resolve dependencies based
    on provided configurations.

    Components:
    - **config**: Stores configuration values for
    clustering algorithms and handlers.
    - **input_handler**: Selects the appropriate input handler
    (`JSONHandler` or `NumpyHandler`)
      based on the input format specified in `config.clustering.input_format`.
    - **output_handler**: Selects the appropriate output handler
    (`JSONHandler` or `NumpyHandler`)
      based on the output format specified in
      `config.clustering.output_format`.
    - **clustering_algorithm**:
    Dynamically initializes the clustering algorithm
      (`CKMeans`, `CDBSCAN`, or `CAgglomerativeClustering`)
      based on the value of
      `config.clustering.algorithm`.
    - **io_handler**: Initializes an `IOHandler`
    with the selected input and output handlers.

    Example Configuration:
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
                "metric_params": None,
                "algorithm": "auto",
                "leaf_size": 30,
                "p": None,
                "n_jobs": -1
            }
        }
    }
    ```

    Example Usage:
    ```python
    from dependency_injector import providers
    from clustering_container import ClusteringContainer
    from numpy import array

    # Define configuration
    config = {
        "clustering": {
            "input_format": "json",
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
            }
        }
    }

    # Initialize the container
    container = ClusteringContainer()
    container.config.from_dict(config)

    # Resolve the clustering algorithm and handlers
    clustering = container.clustering_algorithm()
    io_handler = container.io_handler()

    # Example data
    data = array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

    # Perform clustering
    labels = clustering.cluster(data)
    print("Cluster Labels:", labels)

    # Save the results using IOHandler
    io_handler.save(data, "data_output_path")
    ```

    Notes:
    - The `clustering_algorithm`
    dynamically initializes the chosen algorithm based on
      the `config.clustering.algorithm` setting.
      Hyperparameters for the algorithm are
      passed from the corresponding `config.clustering.hyperparameters`.
    - The `input_handler` and `output_handler`
    allow seamless data handling between
      different formats (e.g., JSON and NumPy arrays).
    - This container promotes loose coupling and
    modularity in the codebase, making it
      easier to switch components (e.g., algorithms or handlers)
      by updating the configuration.
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

    io_handler = providers.Factory(
        IOHandler, input_handler=input_handler, output_handler=output_handler
    )
