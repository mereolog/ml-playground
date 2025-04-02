from dataclasses import dataclass
from typing import Optional

@dataclass
class KMeansParams(UnsupervisedAlgorithmsParams):
    """This model parameters config defines types and possible default arguments used by the KMeans clustering model

    Attributes:
        n_clusters: Number of clusters to form (default: 8)
        max_iter: Maximum number of iterations of the k-means algorithm (default: 300)
        tol: Tolerance to declare convergence (default: 1e-4)
        init: Method for initialization of centroids ('k-means++', 'random') (default: 'random')
        n_init: Number of times the k-means algorithm will be run with different centroid seeds (default: 10)
        metric: Distance metric to use ('euclidean', 'manhattan') (default: 'euclidean')
    """

    n_clusters: int = 8
    max_iter: int = 300
    tol: float = 1e-4
    init: str = "random"
    n_init: int = 10
    metric: str = "euclidean"


def __post_init__(self):
        super().__post_init__()
