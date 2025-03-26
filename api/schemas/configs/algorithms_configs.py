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
from typing import Literal, Optional  # will need to import other types

# its mostly for type checkers
# now they will warn us if we try to assign invalid string
LossType = Literal["mse", "mae"]

# the Optional type tells us that the regularization is optional and can be equal None
RegType = Optional[Literal["l1", "l2", "elasticnet"]]


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

    def __post_init__(self):
        # validation logic goes here

        if not (0 < self.test_size < 1):
            raise ValueError(
                f"test_size must be between 0 and 1 (exclusive), got {self.test_size}"
            )


@dataclass
class UnsupervisedAlgorithmsParams(BaseAlgorithmParams):
    """Base parameters for all unsupervised algorithms

    Attributes:
        n_init: Number of times the algorithm will be run
        max_iter: Maximum number of iterations for the algorithm
    """

    n_init: int = 10
    max_iter: int = 300

    def __post_init__(self):
        pass


@dataclass
class LinearRegressionParams(SupervisedAlgorithmsParams):
    """This model parameters config defines types and possible defualt arguments used by LinearRegression model

    Attributes:
        learning_rate: Step size for gradient descent optimization (default: 0.01)
        epochs: Number of complete passes through the training dataset (default: 100)
        regularization: L2 regularization strength to prevent overfitting (default: None)
        batch_size: Number of samples per gradient update, None means full batch (default: None)

        loss: Loss function to use ('mse', 'mae') (default: 'mse')

        reg_type: Regularization type to use ('l1', 'l2', 'elasticnet') (default: None)
        reg_strenght: Strength (lambda/alpha) of the regularization. Must be > 0 to have an effect.
        mixing_ratio: Mixing parameter for ElasticNet regularization. Must be 0 <= mixing_ratio <= 1.
                      0.0 value corresponds to L2 only, and 1.0 to L1 only.
                      Only used if reg_type='elasticnet'. (default: 0.5)
    """

    learning_rate: float = 0.01
    epochs: int = 100
    regularization: Optional[float] = None
    batch_size: Optional[int] = None

    # -- loss function config --
    loss: LossType = "mse"

    # -- regularization config --
    reg_type: RegType = None
    reg_strenght: float = 0.01
    mixing_ratio: float = 0.5  # default mixing ratio for ElasticNet

    def __post_init__(self):
        super().__post_init__()
        # validation goes here, we also have to run init on parent class


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
