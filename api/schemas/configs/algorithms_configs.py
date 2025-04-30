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

import sys
from dataclasses import dataclass
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


# --- NEW Generalized Class for Gradient-Based/Iterative Parameters ---
@dataclass
class GradientBasedParams(SupervisedAlgorithmsParams):
    """
    General parameters for supervised algorithms using iterative optimization
    like Gradient Descent. Inherits SupervisedAlgorithmsParams.

    Attributes:
        learning_rate: Step size for gradient optimization (default: 0.01).
        epochs: Number of full passes through the training dataset (default: 100).
        batch_size: Number of samples per gradient update, None means the entire dataset (full batch) (default: None).
        reg_type: Type of regularization ('l1', 'l2', 'elasticnet') or None (default: None).
        reg_strength: Strength (lambda/alpha) of the regularization. Must be > 0 to have an effect (default: 0.01).
                      (Corrected typo from reg_strenght)
        mixing_ratio: Mixing parameter for ElasticNet regularization. Must be 0 <= mixing_ratio <= 1.
                      A value of 0.0 corresponds to L2 only, and 1.0 to L1 only.
                      Used only when reg_type='elasticnet' (default: 0.5).
    """

    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: Optional[int] = None

    # -- Regularization Configuration --
    reg_type: RegType = None
    reg_strength: float = 0.01  # Corrected typo
    mixing_ratio: float = 0.5

    def __post_init__(self):
        """Validate parameters after initialization."""
        super().__post_init__()  # Call parent __post_init__ (includes Supervised and Base validation)

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer or None")

        if self.reg_type is not None:
            if self.reg_strength <= 0:
                raise ValueError(
                    "reg_strength must be greater than 0 when regularization is used (reg_type is not None)"
                )

            if self.reg_type == "elasticnet":
                if not (0.0 <= self.mixing_ratio <= 1.0):
                    raise ValueError(
                        "mixing_ratio must be between 0 and 1 for ElasticNet"
                    )
            elif self.mixing_ratio != 0.5:
                print(
                    f"Warning: parameter 'mixing_ratio' ({self.mixing_ratio}) is set, "
                    f"but it only has an effect when reg_type='elasticnet'. Current type: {self.reg_type}",
                    file=sys.stderr,
                )
