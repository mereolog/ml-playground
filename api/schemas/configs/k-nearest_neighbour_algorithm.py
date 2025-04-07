"""
Model parameter configurations

This module defines dataclasses for ML algorithm hyperparameters to be used throughout the platform.
These dataclasses provide type safety, default values, and documentation for each algorithm's
configurable parameters.

Most of the stuff assumes that there is some existing logic in our code so we just need to build on top of that.

Usage:
from configs.model_parameter_configs import LinearRegressionParams, KNeighborsParams # Added KNeighborsParams

# creates a linear regression model config with default values
params = LinearRegressionParams()

# creates config but with custom values
custom_params = LinearRegressionParams(learning_rate=0.05, epochs=200)

# creates a KNN model config with default values
knn_params = KNeighborsParams()

# creates KNN config but with custom values
custom_knn_params = KNeighborsParams(n_neighbors=3, weights='distance')


Dataclasses documentation:
https://docs.python.org/3/library/dataclasses.html
"""
from dataclasses import dataclass, field # 'field' is imported but not used in this example
from typing import Literal, Optional, Union, Dict, Any # Added Dict, Any for potential metric_params

# --- Type Definitions ---
# It's mostly for type checkers;
# now they will warn us if we try to assign invalid string
LossType = Literal["mse", "mae"]
# The Optional type tells us that the regularization is optional and can be equal to None
RegType = Optional[Literal["l1", "l2", "elasticnet"]]

# Type definitions specific to KNN
KNNWeightType = Literal["uniform", "distance"]
KNNAlgorithmType = Literal["auto", "ball_tree", "kd_tree", "brute"]
KNNMetricType = Literal[
    "euclidean", "manhattan", "minkowski", "chebyshev",
    # Other metrics supported by libraries like scikit-learn could be added
]

# --- Base Parameter Classes ---
@dataclass
class BaseAlgorithmParams:
    """Base parameters common to all ML models.

    This class provides common configuration parameters that are relevant
    across different algorithm types.

    Attributes:
        random_state: Seed for random number generation (for reproducibility).
        verbose: Flag to control logging verbosity.
    """

    random_state: Optional[int] = None
    verbose: bool = False

@dataclass
class SupervisedAlgorithmsParams(BaseAlgorithmParams):
    """Base parameters for all supervised algorithms. Consider the most common parameters.

    Attributes:
       test_size: Proportion of the dataset to include in the test split.
       validation_size: Proportion of the TRAINING DATA to use as validation.
       shuffle: Whether to shuffle the dataset before splitting.
       stratify: Whether to stratify the datasets based on the target values (y).
                Often useful for classification tasks.
    """

    test_size: float = 0.2
    # Optional[...] is basically Union[..., None]
    # It's a prettier way of telling our script that we expect the argument to be a float or None value
    validation_size: Optional[float] = (
        None  # We might not want to create a validation split, so we default to None
    )
    shuffle: bool = True
    stratify: bool = False # For classification, stratify=True might be a better default

    def __post_init__(self):
        # Validation logic goes here
        if not (0 < self.test_size < 1):
            raise ValueError(
                f"test_size must be between 0 and 1 (exclusive), got {self.test_size}"
            )
        if self.validation_size is not None and not (0 < self.validation_size < 1):
             raise ValueError(
                f"validation_size must be None or between 0 and 1 (exclusive), got {self.validation_size}"
            )


@dataclass
class UnsupervisedAlgorithmsParams(BaseAlgorithmParams):
    """Base parameters for all unsupervised algorithms.

    Attributes:
        n_init: Number of times the algorithm will be run with different centroid seeds.
                The final results will be the best output of n_init consecutive runs
                in terms of inertia (for K-Means).
        max_iter: Maximum number of iterations of the algorithm for a single run.
    """

    n_init: int = 10
    max_iter: int = 300

    def __post_init__(self):
       # Validation logic for unsupervised parameters
       if self.n_init <= 0:
           raise ValueError(f"n_init must be positive, got {self.n_init}")
       if self.max_iter <= 0:
           raise ValueError(f"max_iter must be positive, got {self.max_iter}")

# --- Algorithm Specific Parameter Classes ---

@dataclass
class LinearRegressionParams(SupervisedAlgorithmsParams):
    """Parameters config defining types and defaults for a Linear Regression model
       (likely one implemented with gradient descent).

    Attributes:
        learning_rate: Step size for gradient descent optimization (default: 0.01).
        epochs: Number of complete passes through the training dataset (default: 100).
        # regularization: Optional[float] = None # Seems redundant with reg_type/reg_strength below. Consider removing.
        batch_size: Number of samples per gradient update. None means full batch (default: None).

        loss: Loss function to use ('mse', 'mae') (default: 'mse').

        reg_type: Regularization type to use ('l1', 'l2', 'elasticnet') (default: None).
        reg_strength: Strength (lambda/alpha) of the regularization. Must be > 0 to have an effect (default: 0.01).
        mixing_ratio: Mixing parameter for ElasticNet regularization (L1 ratio). Must be 0 <= mixing_ratio <= 1.
                      0.0 corresponds to L2 only, 1.0 to L1 only.
                      Only used if reg_type='elasticnet' (default: 0.5).
    """
    learning_rate: float = 0.01
    epochs: int = 100
    # regularization: Optional[float] = None # Commented out as likely redundant
    batch_size: Optional[int] = None

    # -- loss function config --
    loss: LossType = "mse"

    # -- regularization config --
    reg_type: RegType = None
    reg_strength: float = 0.01 # Corrected spelling from 'strenght'
    mixing_ratio: float = 0.5  # Default mixing ratio for ElasticNet

    def __post_init__(self):
        """Post-initialization validation for Linear Regression parameters."""
        super().__post_init__() # Call validation from the parent class
        # Validation specific to Linear Regression
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size is not None and self.batch_size <= 0:
             raise ValueError(f"batch_size must be None or positive, got {self.batch_size}")
        if self.reg_type is not None and self.reg_strength <= 0:
             # Raise error if user *intends* regularization but strength is invalid
             raise ValueError(f"reg_strength must be positive when reg_type is set, got {self.reg_strength}")
        if self.reg_type == "elasticnet" and not (0 <= self.mixing_ratio <= 1):
             raise ValueError(f"mixing_ratio must be between 0 and 1 for elasticnet, got {self.mixing_ratio}")


@dataclass
class DecisionTreeParams(SupervisedAlgorithmsParams):
    """Parameters for Decision Tree Classifier/Regressor.

    Attributes:
        criterion: Function to measure the quality of a split.
                   For Classification: 'gini', 'entropy', 'log_loss'.
                   For Regression: 'squared_error', 'absolute_error', 'friedman_mse', 'poisson'.
                   (default: 'gini')
        max_depth: Maximum depth of the tree. If None, nodes are expanded until all leaves are pure
                   or contain less than min_samples_split samples. (default: None).
        min_samples_split: Minimum number of samples required to split an internal node (default: 2).
        min_samples_leaf: Minimum number of samples required to be at a leaf node (default: 1).
        # Other parameters like max_features, class_weight etc. can be added here.
    """
    criterion: str = "gini" # Default for classification; change if primarily used for regression
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    # Other params like: max_features: Optional[Union[int, float, Literal["auto", "sqrt", "log2"]]] = None

    def __post_init__(self):
        """Post-initialization validation for Decision Tree parameters."""
        super().__post_init__()
        # Validation specific to Decision Tree
        if self.max_depth is not None and self.max_depth <= 0:
             raise ValueError(f"max_depth must be None or positive, got {self.max_depth}")
        # Check min_samples_split based on type (int or float) if float is allowed
        if isinstance(self.min_samples_split, int) and self.min_samples_split <= 1:
             raise ValueError(f"min_samples_split (int) must be >= 2, got {self.min_samples_split}")
        # Similar check for min_samples_leaf
        if isinstance(self.min_samples_leaf, int) and self.min_samples_leaf <= 0:
             raise ValueError(f"min_samples_leaf (int) must be positive, got {self.min_samples_leaf}")
        # Add checks for float types if they are allowed (e.g., represent fractions)


# ---- HERE IS THE CODE FOR K-NEAREST NEIGHBORS ----
@dataclass
class KNeighborsParams(SupervisedAlgorithmsParams):
    """Parameters for K-Nearest Neighbors algorithm (Classifier/Regressor).

    Attributes:
        n_neighbors: Number of neighbors to use by default for kneighbors queries (default: 5).
        weights: Weight function used in prediction. Possible values:
                 - 'uniform' : All points in each neighborhood are weighted equally.
                 - 'distance' : weight points by the inverse of their distance. Closer neighbors
                                have greater influence.
                 (default: 'uniform')
        algorithm: Algorithm used to compute the nearest neighbors:
                   - 'ball_tree' will use BallTree
                   - 'kd_tree' will use KDTree
                   - 'brute' will use a brute-force search.
                   - 'auto' will attempt to decide the most appropriate algorithm
                     based on the values passed to fit method.
                   (default: 'auto')
        leaf_size: Leaf size passed to BallTree or KDTree. This can affect the speed
                   of construction and query, as well as the memory required to store the tree.
                   (default: 30)
        p: Power parameter for the Minkowski metric.
           When p = 1, this is equivalent to using manhattan_distance (l1).
           When p = 2, this is equivalent to using euclidean_distance (l2).
           For arbitrary p, minkowski_distance (l_p) is used. (default: 2)
        metric: The distance metric to use for the tree.
                Common values: 'euclidean', 'manhattan', 'minkowski', 'chebyshev'.
                (default: 'minkowski')
        metric_params: Additional keyword arguments for the metric function (default: None).
        # n_jobs: Number of parallel jobs to run for neighbors search. -1 means using all processors.
        #         Might belong in a higher-level execution config. (default: None)
    """
    n_neighbors: int = 5
    weights: KNNWeightType = "uniform"
    algorithm: KNNAlgorithmType = "auto"
    leaf_size: int = 30
    p: int = 2  # Default for Euclidean distance (when metric='minkowski')
    metric: KNNMetricType = "minkowski" # Minkowski with p=2 is Euclidean
    metric_params: Optional[Dict[str, Any]] = None # Optional dict for metric params
    # n_jobs: Optional[int] = None

    def __post_init__(self):
        """Post-initialization validation for KNN parameters."""
        super().__post_init__() # Call validation from the parent class

        # Validation specific to KNN
        if self.n_neighbors <= 0:
            raise ValueError(f"n_neighbors must be positive, got {self.n_neighbors}")

        if self.leaf_size <= 0:
            raise ValueError(f"leaf_size must be positive, got {self.leaf_size}")

        # Although `p` is technically used only with the Minkowski metric by default,
        # libraries often expect it to be >= 1.
        if self.p < 1:
             raise ValueError(f"p must be >= 1 for Minkowski metric, got {self.p}")

        # Optional: More complex validation could be added, e.g.,
        # checking if `p` is relevant only when `metric='minkowski'`,
        # but the above validation is simpler and often sufficient.


@dataclass
class KMeansParams(UnsupervisedAlgorithmsParams):
    """Parameters for K-Means clustering algorithm.

    Attributes:
        n_clusters: The number of clusters to form as well as the number of centroids to generate.
                    This is the 'K' in K-Means. (default: 8 - but should often be tuned)
        init: Method for initialization ('k-means++', 'random').
              'k-means++' selects initial cluster centers in a smart way to speed up convergence.
              'random' chooses n_clusters observations (rows) at random from data for the initial centroids.
              (default: 'k-means++')
        tol: Relative tolerance with regards to Frobenius norm of the difference
             in the cluster centers of two consecutive iterations to declare convergence.
             (default: 1e-4)
        # algorithm: K-means algorithm to use. 'lloyd' is the standard, 'elkan' can be faster
        #            on data with well-defined clusters, but uses more memory.
        #            'auto' chose 'elkan' in older sklearn, now typically defaults to 'lloyd'.
        #            (default might vary by library version, e.g., 'lloyd')
    """
    n_clusters: int = 8 # K in K-Means, must typically be defined or tuned
    init: Literal['k-means++', 'random'] = 'k-means++'
    tol: float = 1e-4
    # algorithm: Literal['lloyd', 'elkan', 'auto'] = 'lloyd' # Name might depend on lib version

    def __post_init__(self):
        """Post-initialization validation for K-Means parameters."""
        super().__post_init__() # Call validation (n_init, max_iter) from the parent class

        # Validation specific to K-Means
        if self.n_clusters <= 0:
            raise ValueError(f"n_clusters must be positive, got {self.n_clusters}")
        if self.tol < 0:
            raise ValueError(f"tol must be non-negative, got {self.tol}")


# Example usage (outside class definitions)
if __name__ == "__main__":
    print("--- KNN Examples ---")
    default_knn = KNeighborsParams()
    print(f"Default KNN params: {default_knn}")

    custom_knn = KNeighborsParams(
        n_neighbors=3,
        weights='distance',
        metric='manhattan', # p is ignored if metric is not 'minkowski' in some implementations
        p=1, # Explicitly setting p=1 for Manhattan metric consistency
        test_size=0.3,
        stratify=True, # Good idea for classification
        random_state=42
    )
    print(f"Custom KNN params: {custom_knn}")

    try:
        invalid_knn = KNeighborsParams(n_neighbors=0)
    except ValueError as e:
        print(f"\nCaught expected validation error: {e}")

    try:
        invalid_knn_p = KNeighborsParams(metric='minkowski', p=0)
    except ValueError as e:
        print(f"Caught expected validation error for p: {e}")

    print("\n--- K-Means Example ---")
    # Note: n_clusters usually requires thoughtful selection or tuning
    default_kmeans = KMeansParams(n_clusters=5) # n_clusters is essential for K-Means
    print(f"Default KMeans params (with n_clusters=5): {default_kmeans}")

    try:
        invalid_kmeans = KMeansParams(n_clusters=0, n_init=10, max_iter=100)
    except ValueError as e:
        print(f"\nCaught expected validation error for K-Means: {e}")

    print("\n--- Decision Tree Example ---")
    dt_params = DecisionTreeParams(criterion='entropy', max_depth=10)
    print(f"Decision Tree params: {dt_params}")

    print("\n--- Linear Regression Example ---")
    lr_params = LinearRegressionParams(reg_type='l2', reg_strength=0.05)
    print(f"Linear Regression params: {lr_params}")