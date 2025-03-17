"""
Model parameter configurations

This module defines dataclasses for ML algorithm hyperparameters to be used throughout the platform.
These dataclasses provide type safety, default values, and documentation for each algorithm's
configurable parameters.

Usage:
    from configs.model_parameter_configs import LinearRegressionParams
    
    # creates a linear regression model config with default values
    params = LinearRegressionParams()
    
    # creates config but with custom values
    custom_params = LinearRegressionParams(learning_rate=0.05, epochs=200)
"""

from dataclasses import dataclass, field
from typing import Optional  # will need to import other types


@dataclass
class LinearRegressionParams:
    """This model parameters config defines types and possible defualt arguments used by LinearRegression model

    Attributes:
        learning_rate: Step size for gradient descent optimization (default: 0.01)
        epochs: Number of complete passes through the training dataset (default: 100)
        regularization: L2 regularization strength to prevent overfitting (default: None)
        batch_size: Number of samples per gradient update, None means full batch (default: None)
    """

    learning_rate: float = 0.01
    epochs: int = 100
    # Optional[...] is basically Union[..., None]
    # Its a prettier way of telling our script that we expect the argument to be float or None
    regularization: Optional[float] = None
    batch_size: Optional[int] = None


@dataclass
class DecisionTreeParams:
    pass


@dataclass
class KMeansParams:
    pass


# and the rest of the dataclasses for our models
# think about what parameters will your algorithm need to consume to work
