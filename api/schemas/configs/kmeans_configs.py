from dataclasses import dataclass
from typing import Optional, Literal
from schemas.configs.algorithms_configs import UnsupervisedAlgorithmsParams

@dataclass
class KMeansParams(UnsupervisedAlgorithmsParams):
    """This model parameters config defines types and possible default arguments used by the KMeans clustering model

    Attributes:
        n_clusters: Number of clusters to form (default: 8)
        max_iter: Maximum number of iterations of the k-means algorithm (default: 300)
        tol: Tolerance to declare convergence (default: 1e-4)
        init_method: Method for initialization of centroids ('k-means++', 'random') (default: 'random')
        initialization_runs: Number of times the k-means algorithm will be run with different centroid seeds (default: 10)
        metric: Distance metric to use ('euclidean', 'manhattan') (default: 'euclidean')
    """

    n_clusters: int = 8
    max_iter: int = 300
    tol: float = 1e-4
    init_method: Literal["k-means++", "random"] = "random"
    initialization_runs: int = 10
    metric: Literal["euclidean", "manhattan"] = "euclidean"


    def __post_init__(self):
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if self.tol <= 0:
            raise ValueError("tol must be a positive float")
        if self.init_method not in {"k-means++", "random"}:
            raise ValueError("init must be 'k-means++' or 'random'")
        if self.initialization_runs <= 0:
            raise ValueError("n_init must be a positive integer")
        if self.metric not in {"euclidean", "manhattan"}:
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
