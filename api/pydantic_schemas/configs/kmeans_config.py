from typing import Literal

from pydantic import Field
from pydantic_schemas.configs.algorithm_configs import UnsupervisedAlgorithmsParams

# Define custom types for clarity, mirroring the dataclass version
InitMethodType = Literal["k-means++", "random"]
MetricType = Literal["euclidean", "manhattan"]


class KMeansParams(UnsupervisedAlgorithmsParams):
    """
    Pydantic model for K-Means clustering algorithm parameters.

    Attributes:
        n_clusters: Number of clusters to form (default: 8). Must be positive.
        max_iter: Maximum number of iterations of the k-means algorithm (default: 300). Must be positive.
        tol: Tolerance to declare convergence (default: 1e-4). Must be positive.
        init_method: Method for initialization of centroids ('k-means++', 'random') (default: 'random').
        initialization_runs: Number of times the k-means algorithm will be run with different centroid seeds (default: 10). Must be positive.
        metric: Distance metric to use ('euclidean', 'manhattan') (default: 'euclidean').
    """

    n_clusters: int = Field(default=8, description="Number of clusters to form.", gt=0)
    max_iter: int = Field(
        default=300,
        description="Maximum number of iterations of the k-means algorithm for a single run.",
        gt=0,
    )
    tol: float = Field(
        default=1e-4,
        description="Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.",
        gt=0,
    )
    init_method: InitMethodType = Field(
        default="random",
        description="Method for initialization: 'k-means++' : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. 'random': choose n_clusters observations (rows) at random from data for the initial centroids.",
    )
    initialization_runs: int = Field(
        default=10,
        description="Number of times the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.",
        gt=0,
    )
    metric: MetricType = Field(
        default="euclidean",
        description="Distance metric to use. 'euclidean' or 'manhattan'.",
    )
