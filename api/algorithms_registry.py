from typing import Dict, Any, Optional

from schemas.interface.algorithm_interface import AlgorithmInfo
from schemas.interface.algorithm_interface import AlgorithmRegistryEntry

from schemas.configs.decision_tree_config import DecisionTreeParams
from schemas.configs.k_nearest_neighbour_config import KNeighborsParams
from schemas.configs.kmeans_config import KMeansParams
from schemas.configs.linear_regression_configs import LinearRegressionParams
from schemas.configs.logistic_regression_config import LogisticRegressionParams
from schemas.configs.polynomial_regression_configs import PolynomialRegressionParams

from algorithms.supervised.linear_regression import LinearRegression

ALGORITHM_REGISTRY: Dict[str, AlgorithmRegistryEntry] = {
    "linear_regression": AlgorithmRegistryEntry(
        AlgorithmInfo(
            internal_name="linear_regression",
            display_name="Linear Regression",
            description="A simple linear regression model.",
        ),
        LinearRegressionParams,
        LinearRegression,
    ),
    # Add other algorithms as they're implemented
    "decision_tree": AlgorithmRegistryEntry(
        AlgorithmInfo(
            internal_name="decision_tree",
            display_name="Decision Tree",
            description="A decision tree algorithm for classification and regression.",
        ),
        DecisionTreeParams,
        None,  # Not implemented yet
    ),
    "k_nearest_neighbours": AlgorithmRegistryEntry(
        AlgorithmInfo(
            internal_name="k_nearest_neighbours",
            display_name="K-Nearest Neighbours",
            description=(
                "A k-nearest neighbours algorithm for classification and regression."
            ),
        ),
        KNeighborsParams,
        None,  # Not implemented yet
    ),
    "kmeans": AlgorithmRegistryEntry(
        AlgorithmInfo(
            internal_name="kmeans",
            display_name="K-Means Clustering",
            description="A k-means clustering algorithm to partition data into k clusters.",
        ),
        KMeansParams,
        None,  # Not implemented yet
    ),
    "logistic_regression": AlgorithmRegistryEntry(
        AlgorithmInfo(
            internal_name="logistic_regression",
            display_name="Logistic Regression",
            description="A logistic regression algorithm for binary classification.",
        ),
        LogisticRegressionParams,
        None,  # Not implemented yet
    ),
    "polynomial_regression": AlgorithmRegistryEntry(
        AlgorithmInfo(
            internal_name="polynomial_regression",
            display_name="Polynomial Regression",
            description="A polynomial regression algorithm for non-linear relationships.",
        ),
        PolynomialRegressionParams,
        None,  # Not implemented yet
    ),
}