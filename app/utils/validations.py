from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class KMeansConfig(BaseModel):
    """
    Configuration for the KMeans clustering algorithm.

    Attributes:
        n_clusters (int): The number of clusters to form. Default is 8.
        init (Union[Literal["k-means++", "random"], list]):
        Method for initialization.
            - "k-means++" (default): Smart seeding.
            - "random": Random initialization.
            - A custom list for initial centroids.
        n_init (Union[Literal["auto"], int]):
        Number of initializations to perform.
            - "auto" (default): Automatically chooses optimal n_init.
            - Integer: Explicit number of initializations.
        max_iter (int): Maximum number of iterations for a single run.
        Default is 300.
        tol (float): Tolerance for convergence. Default is 1e-4.
        verbose (int): Verbosity level. Default is 0.
        random_state (Optional[int]): Random seed for reproducibility.
        Default is None.
        copy_x (bool): If True (default), data is copied before clustering.
        algorithm (Literal["lloyd", "elkan"]): Algorithm variant to use.
        Default is "lloyd".

    Example usage:
        >>> config = KMeansConfig(n_clusters=5, max_iter=200)
        >>> print(config.dict())
    """

    n_clusters: int = 8
    init: Union[Literal["k-means++", "random"], list] = "k-means++"
    n_init: Union[Literal["auto"], int] = "auto"
    max_iter: int = 300
    tol: float = 1e-4
    verbose: int = 0
    random_state: Optional[int] = None
    copy_x: bool = True
    algorithm: Literal["lloyd", "elkan"] = "lloyd"

    model_config = ConfigDict(extra="forbid")


class DBSCANConfig(BaseModel):
    """
    Configuration for the DBSCAN clustering algorithm.

    Attributes:
        eps (float):
        Maximum distance between samples for clustering. Default is 0.5.
        min_samples (int): Minimum number of samples in a neighborhood
        for a core point. Default is 5.
        metric (str): Distance metric to use. Default is "euclidean".
        metric_params (Optional[dict[str, Any]]):
        Additional metric-specific parameters. Default is None.
        algorithm (Literal["auto", "ball_tree", "kd_tree", "brute"]):
        Algorithm to compute nearest neighbors. Default is "auto".
        leaf_size (int): Leaf size for tree-based algorithms.
        Default is 30.
        p (Optional[float]): Power parameter for Minkowski metric.
        Default is None.
        n_jobs (Optional[int]): Number of parallel jobs to run.
        Default is None.

    Example usage:
        >>> config = DBSCANConfig(eps=0.3, min_samples=10)
        >>> print(config.dict())
    """

    eps: float = 0.5
    min_samples: int = 5
    metric: str = "euclidean"
    metric_params: Optional[dict[str, Any]] = None
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: int = 30
    p: Optional[float] = None
    n_jobs: Optional[int] = None

    model_config = ConfigDict(extra="forbid")


class AgglomerativeConfig(BaseModel):
    """
    Configuration for the Agglomerative Clustering algorithm.

    Attributes:
        n_clusters (Optional[int]):
        Number of clusters to find. Default is 2.
        metric (str):
        Distance metric to use. Default is "euclidean".
        memory (Optional[str]):
        Path to cache computations. Default is None.
        connectivity (Optional[list[Any]]):
        Connectivity matrix. Default is None.
        compute_full_tree (Union[Literal["auto"], bool]):
        Whether to compute the full tree. Default is "auto".
        linkage (Literal["ward", "complete", "average", "single"]):
        Linkage criterion to use. Default is "ward".
        distance_threshold (Optional[float]):
        Threshold to form clusters. Default is None.
        compute_distances (bool):
        Whether to compute distances. Default is False.

    Example usage:
        >>> config = AgglomerativeConfig(n_clusters=3, linkage="average")
        >>> print(config.dict())
    """

    n_clusters: Optional[int] = 2
    metric: str = "euclidean"
    memory: Optional[str] = None
    connectivity: Optional[list[Any]] = None
    compute_full_tree: Union[Literal["auto"], bool] = "auto"
    linkage: Literal["ward", "complete", "average", "single"] = "ward"
    distance_threshold: Optional[float] = None
    compute_distances: bool = False

    model_config = ConfigDict(extra="forbid")


class ClusteringConfig(BaseModel):
    """
    Top-level configuration for clustering operations.

    Attributes:
        input_format (Literal["json", "numpy"]):
        Input data format. Default is "json".
        output_format (Literal["json", "numpy"]):
        Output data format. Default is "json".
        algorithm (Literal["KMeans", "DBSCAN", "AgglomerativeClustering"]):
        Clustering algorithm to use. Default is "KMeans".
        hyperparameters (Union[KMeansConfig, DBSCANConfig,
        AgglomerativeConfig]): Algorithm-specific hyperparameters.

    Validation:
        Ensures that the `hyperparameters`
        field matches the selected
        `algorithm`.

    Example usage:
        >>> config = ClusteringConfig(
        ...     input_format="json",
        ...     output_format="numpy",
        ...     algorithm="DBSCAN",
        ...     hyperparameters={
        ...         "eps": 0.3,
        ...         "min_samples": 10
        ...     }
        ... )
        >>> print(config.dict())
    """

    input_format: Literal["json", "numpy"] = "json"
    output_format: Literal["json", "numpy"] = "json"
    algorithm: Literal[
        "KMeans", "DBSCAN", "AgglomerativeClustering"
    ] = "KMeans"
    hyperparameters: Union[
        KMeansConfig, DBSCANConfig, AgglomerativeConfig
    ] = Field(...)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    def match_hyperparameters_with_algorithm(cls, values: dict):
        """
        Validate that the hyperparameters match the selected algorithm.

        Args:
            values (dict): Input dictionary of configuration values.

        Returns:
            dict: Updated configuration with validated hyperparameters.

        Raises:
            ValueError: If hyperparameters are incompatible with the algorithm.

        Example:
            >>> config = ClusteringConfig(
            ...     algorithm="AgglomerativeClustering",
            ...     hyperparameters={
            ...         "n_clusters": 3,
            ...         "linkage": "average"
            ...     }
            ... )
        """
        algorithm = values.get("algorithm")
        hyperparameters: dict = values.get("hyperparameters", {})

        try:
            if algorithm == "KMeans":
                values["hyperparameters"] = KMeansConfig(**hyperparameters)
            elif algorithm == "DBSCAN":
                values["hyperparameters"] = DBSCANConfig(**hyperparameters)
            elif algorithm == "AgglomerativeClustering":
                if (
                    hyperparameters.get("distance_threshold", None) is not None
                    and hyperparameters.get("n_clusters", None) is not None
                ):
                    raise ValueError(
                        """
                        Both 'distance_threshold'
                        and 'n_clusters'
                        cannot be set simultaneously.
                        """
                    )
                values["hyperparameters"] = AgglomerativeConfig(
                    **hyperparameters
                )
            else:
                raise ValueError(
                    f"""
                    Unsupported algorithm '{algorithm}'.
                    Supported algorithms are KMeans, DBSCAN,
                    and AgglomerativeClustering.
                    """
                )
        except ValueError as e:
            raise ValueError(
                f"""
                Error configuring hyperparameters
                for algorithm '{algorithm}': {e}
                """
            ) from e

        return values


class Config(BaseModel):
    """
    Root configuration class for the application.

    Attributes:
        clustering (ClusteringConfig): Clustering configuration.

    Example usage:
        >>> config = Config(
        ...     clustering={
        ...         "input_format": "json",
        ...         "output_format": "numpy",
        ...         "algorithm": "KMeans",
        ...         "hyperparameters": {"n_clusters": 5}
        ...     }
        ... )
        >>> print(config.dict())
    """

    clustering: ClusteringConfig

    model_config = ConfigDict(extra="forbid")
