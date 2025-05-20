from typing import Literal, Optional

from pydantic import Field
from pydantic_schemas.configs.algorithm_configs import SupervisedAlgorithmsParams

# Define custom types for clarity, mirroring the dataclass version
KNNWeightType = Literal["uniform", "distance"]
KNNAlgorithmType = Literal["auto", "ball_tree", "kd_tree", "brute"]
KNNMetricType = Literal["minkowski", "euclidean", "manhattan", "chebyshev"]


class KNeighborsParams(SupervisedAlgorithmsParams):
    """
    Pydantic model for K-Nearest Neighbors (KNN) algorithm parameters.

    Attributes:
        n_neighbors: Number of neighbors to use (default: 5). Must be positive.
        weights: Weight function used in prediction ('uniform', 'distance').
        algorithm: Algorithm used to compute the nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute').
        leaf_size: Leaf size passed to BallTree or KDTree (default: 30). Must be positive.
        p: Power parameter for the Minkowski metric (default: 2). When p=1, this is equivalent to using manhattan_distance (L1), and euclidean_distance (L2) for p=2. Must be >= 1.
        metric: The distance metric to use for the tree.
    """

    n_neighbors: int = Field(default=5, description="Number of neighbors to use.", gt=0)
    weights: KNNWeightType = Field(
        default="uniform",
        description="Weight function used in prediction. 'uniform' weights all points equally. 'distance' weights points by the inverse of their distance.",
    )
    algorithm: KNNAlgorithmType = Field(
        default="auto", description="Algorithm used to compute the nearest neighbors."
    )
    leaf_size: int = Field(
        default=30,
        description="Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree.",
        gt=0,
    )
    p: int = Field(
        default=2,
        description="Power parameter for the Minkowski metric. When p=1, this is equivalent to using manhattan_distance (L1), and euclidean_distance (L2) for p=2. For arbitrary p, minkowski_distance (Lp) is used.",
        ge=1,
    )
    metric: KNNMetricType = Field(
        default="minkowski",
        description="The distance metric to use for the tree. The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric.",
    )
