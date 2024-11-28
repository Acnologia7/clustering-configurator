from typing import Any, Dict, List, Literal, Optional, Union

from numpy import ndarray
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans


class CKMeans:
    """
    Utility class for KMeans clustering.

    This class simplifies the process of using the KMeans algorithm
    from scikit-learn, providing a wrapper around its implementation.
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
        """
        Initialize the CKMeans class.

        Args:
            n_clusters (int): The number of clusters to form as well as
            the number of centroids to generate.
            init (Union[Literal["k-means++", "random"]]):
            Method for initialization.
                - "k-means++" ensures smarter centroid initialization.
                - "random" selects initial centroids randomly.
            n_init (Union[Literal["auto"], int]):
            Number of times the algorithm
            will be run with different centroid seeds.
            max_iter (int):
            Maximum number of iterations of the KMeans algorithm
            for a single run.
            tol (float): Relative tolerance to declare convergence.
            verbose (int): Verbosity level for debugging. 0 = silent.
            random_state (Optional[int]):
            Seed for reproducibility. `None` for random.
            copy_x (bool): Whether to make a copy of the input data.
            algorithm (Literal["lloyd", "elkan"]):
            KMeans algorithm variant to use.

        Example usage:
            ```python
            from numpy import array
            from your_module import CKMeans

            data = array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
            kmeans = CKMeans(
                n_clusters=2, init="k-means++", n_init="auto",
                max_iter=300, tol=1e-4, verbose=0, random_state=42,
                copy_x=True, algorithm="lloyd"
            )
            labels = kmeans.cluster(data)
            print(labels)
            ```
        """
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
        """
        Perform clustering on the input data and return the cluster labels.

        Args:
            data (Union[ndarray, list[list[float]]]):
            Input data for clustering.
            sample_weight (Optional[List[float]]):
            Sample weights to influence clustering.

        Returns:
            ndarray: Cluster labels for each data point.

        Example usage:
            ```python
            labels = kmeans.cluster(data)
            print(labels)
            ```
        """
        return self.model.fit_predict(data, sample_weight=sample_weight)


class CDBSCAN:
    """
    Utility class for DBSCAN clustering.

    This class simplifies the process of using the DBSCAN algorithm
    from scikit-learn, providing a wrapper around its implementation.
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
        """
        Initialize the CDBSCAN class.

        Args:
            eps (float): The maximum distance between two samples
            for them to be considered as in the same neighborhood.
            min_samples (int): The minimum number of samples in a neighborhood
            for a point to be considered a core point.
            metric (str): The metric to use for distance computation.
            metric_params (Optional[Dict[str, Any]]):
            Additional keyword arguments for the metric function.
            algorithm (Literal["auto", "ball_tree", "kd_tree", "brute"]):
            Algorithm used to compute nearest neighbors.
            leaf_size (int): Leaf size for the tree-based algorithms.
            p (Optional[float]): The power parameter for Minkowski metric.
            n_jobs (Optional[int]): The number of parallel jobs.

        Example usage:
            ```python
            from numpy import array
            from your_module import CDBSCAN

            data = array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
            dbscan = CDBSCAN(
                eps=0.5, min_samples=2, metric="euclidean",
                metric_params=None, algorithm="auto",
                leaf_size=30, p=None, n_jobs=-1
            )
            labels = dbscan.cluster(data)
            print(labels)
            ```
        """
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
        """
        Perform clustering on the input data and return the cluster labels.

        Args:
            data (Union[ndarray, list[list[float]]]):
            Input data for clustering.
            sample_weight (Optional[List[float]]):
            Sample weights to influence clustering.

        Returns:
            ndarray: Cluster labels for each data point.

        Example usage:
            ```python
            labels = dbscan.cluster(data)
            print(labels)
            ```
        """
        return self.model.fit_predict(data, sample_weight=sample_weight)


class CAgglomerativeClustering:
    """
    Utility class for Agglomerative Clustering.

    This class simplifies the process of using the Agglomerative Clustering
    algorithm from scikit-learn, providing a wrapper around its implementation.
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
        """
        Initialize the CAgglomerativeClustering class.

        Args:
            n_clusters (int): The number of clusters to find.
            metric (str): The metric to use for computing distances.
            memory (Optional[str]):
            Cache to use for storing intermediate results.
            connectivity (Optional[list[Any]]):
            Connectivity matrix for samples.
            compute_full_tree (Union[Literal["auto"], bool]):
            Whether to compute the full tree.
            linkage (Literal["ward", "complete", "average", "single"]):
            Linkage criterion to use.
            distance_threshold (Optional[float]):
            Threshold for merging clusters.
            compute_distances (bool):
            Whether to compute distances between clusters.

        Example usage:
            ```python
            from numpy import array
            from your_module import CAgglomerativeClustering

            data = array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
            agglomerative = CAgglomerativeClustering(
                n_clusters=2, metric="euclidean", memory=None,
                connectivity=None, compute_full_tree="auto",
                linkage="ward", distance_threshold=None,
                compute_distances=False
            )
            labels = agglomerative.cluster(data)
            print(labels)
            ```
        """
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

    def cluster(self, data: ndarray) -> ndarray:
        """
        Perform clustering on the input data and return the cluster labels.

        Args:
            data (Union[ndarray, list[list[float]]]):
            Input data for clustering.

        Returns:
            ndarray: Cluster labels for each data point.

        Example usage:
            ```python
            labels = agglomerative.cluster(data)
            print(labels)
            ```
        """
        return self.model.fit_predict(data)
