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
    if self.n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer")
    if self.max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if self.tol <= 0:
        raise ValueError("tol must be a positive float")
    if self.init not in {"k-means++", "random"}:
        raise ValueError("init must be 'k-means++' or 'random'")
    if self.n_init <= 0:
        raise ValueError("n_init must be a positive integer")
    if self.metric not in {"euclidean", "manhattan"}:
        raise ValueError("metric must be 'euclidean' or 'manhattan'")
