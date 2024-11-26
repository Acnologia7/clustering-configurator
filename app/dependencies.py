from clustering import GeneralClusteringWrapper
from dependency_injector import containers, providers
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from utils.io_handler import IOHandler


class ClusteringContainer(containers.DeclarativeContainer):
    """
    Dependency injection container for clustering components.

    Provides configuration-driven instantiation of clustering algorithms
    wrapped in the `GeneralClusteringWrapper` and an IO handler.
    """

    config: providers.Configuration = providers.Configuration()

    clustering_algorithm: providers.Selector = providers.Selector(
        config.clustering.algorithm,
        KMeans=providers.Factory(
            GeneralClusteringWrapper,
            base_class=KMeans,
            config=config,
        ),
        DBSCAN=providers.Factory(
            GeneralClusteringWrapper,
            base_class=DBSCAN,
            config=config,
        ),
        AgglomerativeClustering=providers.Factory(
            GeneralClusteringWrapper,
            base_class=AgglomerativeClustering,
            config=config,
        ),
    )

    io_handler: providers.Factory[IOHandler] = providers.Factory(IOHandler)
