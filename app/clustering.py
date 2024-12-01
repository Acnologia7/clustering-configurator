from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

from numpy import ndarray
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans


class BaseCluster(ABC):
    """
    Abstract base class for clustering algorithms.

    Subclasses must implement the `cluster` method to perform clustering.

    Example:
    ```python
    class CustomCluster(BaseCluster):
        def cluster(self, data: ndarray):
            # Custom clustering logic here
            pass
    ```
    """

    @abstractmethod
    def cluster(
        self, data: ndarray, sample_weight: Optional[List[float]] = None
    ) -> ndarray:
        """
        Perform clustering on the data and return cluster labels.

        Args:
            data (ndarray): Input data for clustering.
            sample_weight (Optional[List[float]]): Weights for the samples.

        Returns:
            ndarray: Cluster labels for each data point.
        """
        pass


class CKMeans(BaseCluster):
    """
    Utility class for KMeans clustering using scikit-learn.

    Args:
        n_clusters (int): The number of clusters to form.
        init (str): Method for initialization ('k-means++' or 'random').
        n_init (int or str): Number of time the k-means algorithm will be run.
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Tolerance to declare convergence.
        verbose (int): Verbosity level.
        random_state (Optional[int]): Random state for reproducibility.
        copy_x (bool): Whether to copy data.
        algorithm (str): Algorithm to use ('lloyd' or 'elkan').

    Example:
    ```python
    kmeans = CKMeans(
        n_clusters=3,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=42,
        copy_x=True,
        algorithm='lloyd'
    )
    labels = kmeans.cluster(data)
    ```
    """

    def __init__(
        self,
        n_clusters: int,
        init: Union[Literal["k-means++", "random"]],
        n_init: Union[Literal["auto"], int],
        max_iter: int,
        tol: float,
        verbose: int,
        random_state: Optional[int],
        copy_x: bool,
        algorithm: Literal["lloyd", "elkan"],
    ) -> None:
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm,
        )

    def cluster(
        self, data: ndarray, sample_weight: Optional[List[float]] = None
    ) -> ndarray:
        return self.model.fit_predict(data, sample_weight=sample_weight)


class CDBSCAN(BaseCluster):
    """
    Utility class for DBSCAN clustering using scikit-learn.

    Args:
        eps (float):
        The maximum distance between two samples for
        them to be considered as in the same neighborhood.
        min_samples (int):
        The number of samples in a neighborhood for
        a point to be considered as a core point.
        metric (str): The distance metric to use.
        metric_params (Optional[Dict[str, Any]]):
        Additional parameters for the metric.
        algorithm (str):
        Algorithm to compute nearest neighbors
        ('auto', 'ball_tree', 'kd_tree', 'brute').
        leaf_size (int): Leaf size for tree-based algorithms.
        p (Optional[float]): Power parameter for Minkowski metric.
        n_jobs (Optional[int]): The number of parallel jobs to run.

    Example:
    ```python
    dbscan = CDBSCAN(
        eps=0.5,
        min_samples=5,
        metric='euclidean',
        algorithm='auto',
        leaf_size=30
    )
    labels = dbscan.cluster(data)
    ```
    """

    def __init__(
        self,
        eps: float,
        min_samples: int,
        metric: str,
        metric_params: Optional[Dict[str, Any]],
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"],
        leaf_size: int,
        p: Optional[float],
        n_jobs: Optional[int],
    ) -> None:
        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs,
        )

    def cluster(
        self, data: ndarray, sample_weight: Optional[List[float]] = None
    ) -> ndarray:
        return self.model.fit_predict(data, sample_weight=sample_weight)


class CAgglomerativeClustering(BaseCluster):
    """
    Utility class for Agglomerative Clustering using scikit-learn.

    Args:
        n_clusters (int):
        The number of clusters to find.
        metric (str):
        The metric used for distance computation.
        memory (Optional[str]):
        Directory to cache computed distances.
        connectivity (Optional[List[Any]]):
        Connectivity matrix for samples.
        compute_full_tree (Union[Literal['auto'], bool]):
        Whether to compute the full tree.
        linkage (str):
        The linkage criterion ('ward', 'complete', 'average', 'single').
        distance_threshold (Optional[float]):
        The linkage threshold.
        compute_distances (bool):
        Whether to compute distances between clusters.

    Example:
    ```python
    agglo = CAgglomerativeClustering(
        n_clusters=3,
        metric='euclidean',
        linkage='ward',
        compute_distances=False
    )
    labels = agglo.cluster(data)
    ```
    """

    def __init__(
        self,
        n_clusters: int,
        metric: str,
        memory: Optional[str],
        connectivity: Optional[list[Any]],
        compute_full_tree: Union[Literal["auto"], bool],
        linkage: Literal["ward", "complete", "average", "single"],
        distance_threshold: Optional[float],
        compute_distances: bool,
    ) -> None:
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=metric,
            memory=memory,
            connectivity=connectivity,
            compute_full_tree=compute_full_tree,
            linkage=linkage,
            distance_threshold=distance_threshold,
            compute_distances=compute_distances,
        )

    def cluster(
        self, data: ndarray, sample_weight: Optional[List[float]] = None
    ) -> ndarray:
        if sample_weight is not None:
            raise NotImplementedError(
                "CAgglomerativeClustering does not support `sample_weight`."
            )

        return self.model.fit_predict(data)
