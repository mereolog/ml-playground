from dataclasses import dataclass
from typing import Optional, Literal

from schemas.configs.algorithm_configs import BaseAlgorithmParams

# Define custom types for clarity
KNNWeightType = Literal['uniform', 'distance']
KNNAlgorithmType = Literal['auto', 'ball_tree', 'kd_tree', 'brute']
KNNMetricType = Literal['minkowski', 'euclidean', 'manhattan', 'chebyshev']

# Base class for supervised algorithm parameters
@dataclass
class SupervisedAlgorithmsParams(BaseAlgorithmParams):
    """Base parameters for all supervised algorithms.
    Attributes:
        test_size: Proportion of the dataset to include in the test split.
        validation_size: Proportion of the TRAINING DATA to use as validation.
        shuffle: Whether to shuffle the dataset before splitting.
        stratify: Whether to stratify the datasets based on the target values (y).
    """
    test_size: float = 0.2
    validation_size: Optional[float] = None
    shuffle: bool = True
    stratify: bool = False

    def __post_init__(self):
        # Validation logic goes here
        if not (0 < self.test_size < 1):
            raise ValueError(f"test_size must be between 0 and 1 (exclusive), got {self.test_size}")
        if self.validation_size is not None and not (0 < self.validation_size < 1):
            raise ValueError(f"validation_size must be None or between 0 and 1 (exclusive), got {self.validation_size}")

# K-Nearest Neighbors parameters
@dataclass
class KNeighborsParams(SupervisedAlgorithmsParams):
    """Parameters for K-Nearest Neighbors algorithm.
    Attributes:
        n_neighbors: Number of neighbors to use (default: 5).
        weights: Weight function used in prediction ('uniform', 'distance').
        algorithm: Algorithm used to compute the nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute').
        leaf_size: Leaf size passed to tree-based algorithms (default: 30).
        p: Power parameter for the Minkowski distance (default: 2).
        metric: The distance metric to use ('minkowski' is default, which uses p for Euclidean distance).
    """
    n_neighbors: int = 5
    weights: KNNWeightType = 'uniform'
    algorithm: KNNAlgorithmType = 'auto'
    leaf_size: int = 30
    p: int = 2  # Default for Euclidean distance (when metric='minkowski')
    metric: KNNMetricType = 'minkowski'  # Minkowski with p=2 is Euclidean

    def __post_init__(self):
        """Post-initialization validation for KNN parameters."""
        super().__post_init__()

        # Validation specific to KNN
        if self.n_neighbors <= 0:
            raise ValueError(f"n_neighbors must be positive, got {self.n_neighbors}")
        if self.p < 1:
            raise ValueError(f"p must be greater than or equal to 1, got {self.p}")
        if self.leaf_size <= 0:
            raise ValueError(f"leaf_size must be positive, got {self.leaf_size}")

        # Validation for `metric` and `p`
        if self.metric == 'minkowski' and self.p < 1:
            raise ValueError(f"p must be >= 1 for Minkowski metric, got {self.p}")