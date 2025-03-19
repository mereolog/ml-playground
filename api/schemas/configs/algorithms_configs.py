"""
Model parameter configurations

This module defines dataclasses for ML algorithm hyperparameters to be used throughout the platform.
These dataclasses provide type safety, default values, and documentation for each algorithm's
configurable parameters.

Most of the stuff assumes that there is some existing logic in our code so we just need to build on top of that.


Usage:
    from configs.model_parameter_configs import LinearRegressionParams
    
    # creates a linear regression model config with default values
    params = LinearRegressionParams()
    
    # creates config but with custom values
    custom_params = LinearRegressionParams(learning_rate=0.05, epochs=200)
    

Dataclasses documentation:
https://docs.python.org/3/library/dataclasses.html
"""

from dataclasses import dataclass, field
from typing import Optional  # will need to import other types


@dataclass
class BaseAlgorithmParams:
    """Base parameters common to all ML models.

    This class provides common configuration parameters that are relevant
    across different algorithm types.

    Attributes:
        random_state: Seed for random number generation (for reproducibility)
        verbose: Flag to control logging verbosity
    """

    random_state: Optional[int] = None
    verbose: bool = False


@dataclass
class SupervisedAlgorithmsParams(BaseAlgorithmParams):
    """Base parameters for all supervised algorithms, we need to think what are the most common parameters

    Attributes:
       test_size: Proportion of the dataset to include in the test split
       validation_size: Proportion of the TRAINING DATA to use as validation
       shuffle: Whether to shuffle the dataset before spliting
       stratify: Whether to use stratify the datasets based on the target values (y)
    """

    test_size: float = 0.2
    # Optional[...] is basically Union[..., None]
    # Its a prettier way of telling our script that we expect the argument to be a float or None value
    validation_size: Optional[float] = (
        None  # we might not want to create validation split so we default to None
    )
    shuffle: bool = True
    stratify: bool = False


@dataclass
class UnsupervisedAlgorithmsParams(BaseAlgorithmParams):
    """Base parameters for all unsupervised algorithms

    Attributes:
        n_init: Number of times the algorithm will be run
        max_iter: Maximum number of iterations for the algorithm
    """

    n_init: int = 10
    max_iter: int = 300


@dataclass
class LinearRegressionParams(SupervisedAlgorithmsParams):
    """This model parameters config defines types and possible defualt arguments used by LinearRegression model

    Attributes:
        learning_rate: Step size for gradient descent optimization (default: 0.01)
        epochs: Number of complete passes through the training dataset (default: 100)
        regularization: L2 regularization strength to prevent overfitting (default: None)
        batch_size: Number of samples per gradient update, None means full batch (default: None)
    """

    learning_rate: float = 0.01
    epochs: int = 100
    regularization: Optional[float] = None
    batch_size: Optional[int] = None


@dataclass
class DecisionTreeParams(SupervisedAlgorithmsParams):
    # here goes your docstring and code
    pass


@dataclass
class KMeansParams(UnsupervisedAlgorithmsParams):
    # same here
    pass


# and the rest of the dataclasses for our models
# think about what parameters will your algorithm need to consume to work


# the dataclass decorator automaticaly adds methods like __init__ and __repr__ to user-defined classes
# it also does other things but we don't care about that


class ExampleWithoutDataclass:
    def __init__(self, some_string: str, some_float: float, some_int: int):
        self.some_string = some_string
        self.some_float = some_float
        self.some_int = some_int


@dataclass
class ExampleWithDataclass:
    some_string: str
    some_float: float
    some_int: int
