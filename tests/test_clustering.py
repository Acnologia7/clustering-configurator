from numpy import array, ndarray

from app.clustering import CDBSCAN, CAgglomerativeClustering, CKMeans


def test_ckmeans():
    """Test CKMeans clustering."""
    data = array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
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
    labels = kmeans.cluster(data)
    assert isinstance(labels, ndarray), "Output labels should be an ndarray."
    assert len(labels) == len(data), "Each data point should have a label."


def test_cdbscan():
    """Test CDBSCAN clustering."""
    data = array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
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
    labels = dbscan.cluster(data)
    assert isinstance(labels, ndarray), "Output labels should be an ndarray."
    assert len(labels) == len(data), "Each data point should have a label."
    assert (
        -1 in labels or len(set(labels)) > 1
    ), "DBSCAN should identify clusters or noise."


def test_cagglomerative_clustering():
    """Test CAgglomerativeClustering clustering."""
    data = array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
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
    labels = agglomerative.cluster(data)
    assert isinstance(labels, ndarray), "Output labels should be an ndarray."
    assert len(labels) == len(data), "Each data point should have a label."
