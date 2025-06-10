from dataclasses import dataclass
from typing import Literal

from schemas.configs.algorithms_configs import SupervisedAlgorithmsParams

KNNWeightType = Literal["uniform", "distance"]
KNNAlgorithmType = Literal["auto", "ball_tree", "kd_tree", "brute"]
KNNMetricType = Literal["minkowski", "euclidean", "manhattan", "chebyshev"]

@dataclass
class KNeighborsParams(SupervisedAlgorithmsParams):
    n_neighbors: int = 5
    weights: KNNWeightType = "uniform"
    algorithm: KNNAlgorithmType = "auto"
    leaf_size: int = 30
    p: int = 2
    metric: KNNMetricType = "minkowski"

    def __post_init__(self) -> None:
        try:
            super().__post_init__()  # type: ignore[attr-defined]
        except AttributeError:
            pass

        if self.n_neighbors <= 0:
            raise ValueError(f"n_neighbors must be positive, got {self.n_neighbors}")

        if self.leaf_size <= 0:
            raise ValueError(f"leaf_size must be positive, got {self.leaf_size}")

        if self.p < 1:
            raise ValueError(f"p must be ≥ 1, got {self.p}")

        if self.metric not in ("minkowski", "euclidean", "manhattan", "chebyshev"):
            raise ValueError(f"Unsupported metric: {self.metric}")

        if self.metric == "minkowski" and self.p < 1:
            raise ValueError(f"p must be ≥ 1 for Minkowski metric, got {self.p}")
