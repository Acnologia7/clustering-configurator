from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

ALLOWED_METHODS_AND_ATTRIBS: Dict[
    str, Dict[str, Union[List[str], List[str]]]
] = {
    "KMeans": {
        "methods": [
            "fit",
            "fit_predict",
            "fit_transform",
            "predict",
            "score",
            "transform",
        ],
        "attributes": [
            "cluster_centers_",
            "labels_",
            "inertia_",
            "n_iter_",
            "n_features_in_",
            "feature_names_in_",
            "",
        ],
    },
    "DBSCAN": {
        "methods": ["fit", "fit_predict"],
        "attributes": [
            "core_sample_indices_",
            "components_",
            "labels_",
            "n_features_in_",
            "feature_names_in_",
            "",
        ],
    },
    "AgglomerativeClustering": {
        "methods": ["fit", "fit_predict"],
        "attributes": [
            "n_clusters_",
            "labels_",
            "n_leaves_",
            "n_connected_components_",
            "n_features_in_",
            "feature_names_in_",
            "children_",
            "distances_",
            "",
        ],
    },
}


class KMeansConfig(BaseModel):
    n_clusters: int = 8
    init: Union[Literal["k-means++", "random"], list] = "k-means++"
    n_init: Union[Literal["auto"], int] = "auto"
    max_iter: int = 300
    tol: float = 1e-4
    verbose: int = 0
    random_state: Optional[int] = None
    copy_x: bool = True
    algorithm: Literal["lloyd", "elkan"] = "lloyd"

    class Config:
        extra = "forbid"


class DBSCANConfig(BaseModel):
    eps: float = 0.5
    min_samples: int = 5
    metric: str = "euclidean"
    metric_params: Optional[dict[str, Any]] = None
    algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto"
    leaf_size: int = 30
    p: Optional[float] = None
    n_jobs: Optional[int] = None

    class Config:
        extra = "forbid"


class AgglomerativeConfig(BaseModel):
    n_clusters: Optional[int] = 2
    metric: str = "euclidean"
    memory: Optional[str] = None
    connectivity: Optional[list[Any]] = None
    compute_full_tree: Union[Literal["auto"], bool] = "auto"
    linkage: Literal["ward", "complete", "average", "single"] = "ward"
    distance_threshold: Optional[float] = None
    compute_distances: bool = False

    class Config:
        extra = "forbid"


class ClusteringConfig(BaseModel):
    input_format: Literal["json", "numpy"] = "json"
    output_format: Literal["json", "numpy"] = "json"
    method: str
    attribute: Optional[str] = None
    algorithm: Literal[
        "KMeans", "DBSCAN", "AgglomerativeClustering"
    ] = "KMeans"
    hyperparameters: Union[
        KMeansConfig, DBSCANConfig, AgglomerativeConfig
    ] = Field(...)

    class Config:
        extra = "forbid"

    @model_validator(mode="before")
    def match_hyperparameters_with_algorithm(cls, values: dict):
        algorithm = values.get("algorithm")
        method = values.get("method")
        attrib = values.get("attribute")
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
                    AgglomerativeClustering configuration error:
                    Both 'distance_threshold' and 'n_clusters'
                    cannot be set simultaneously.
                    Please set only one of them.
                    """
                    )
                values["hyperparameters"] = AgglomerativeConfig(
                    **hyperparameters
                )
            else:
                raise ValueError(
                    f"""
                    Unsupported algorithm '{algorithm}'.
                    Supported algorithms are:
                    {', '.join(ALLOWED_METHODS_AND_ATTRIBS.keys())}.
                    """
                )
        except ValueError as e:
            raise ValueError(
                f"""
                Error configuring hyperparameters
                for algorithm '{algorithm}': {e}
                """
            ) from e

        allowed_methods: List[str] = ALLOWED_METHODS_AND_ATTRIBS[algorithm][
            "methods"
        ]

        allowed_attr: List[str] = ALLOWED_METHODS_AND_ATTRIBS[algorithm][
            "attributes"
        ]

        if method not in allowed_methods:
            raise ValueError(
                f"""
                Invalid method '{method}'
                for algorithm '{algorithm}'.
                Allowed methods are:
                {', '.join(allowed_methods)}.
                Please check your configuration.
                """
            )

        if attrib not in allowed_attr:
            raise ValueError(
                f"""
                Invalid attribute '{attrib}'
                for algorithm '{algorithm}'.
                Allowed attributes are: {allowed_attr}.
                Please check your configuration.
                """
            )

        return values


class Config(BaseModel):
    clustering: ClusteringConfig

    class Config:
        extra = "forbid"
