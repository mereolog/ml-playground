"""
Model parameter configurations

This module defines classes based on the pydantic models for ML algorithm hyperparameters to be used throughout the platform.
These classes provide type safety, default values, and documentation for each algorithm's
configurable parameters.

Most of the stuff assumes that there is some existing logic in our code so we just need to build on top of that.

Usage:
    from configs.model_parameter_configs import LinearRegressionParams
    
    # creates a linear regression model config with default values
    params = LinearRegressionParams()
    
    # creates config but with custom values
    custom_params = LinearRegressionParams(learning_rate=0.05, epochs=200)
    

Pydantic models documentation:
https://docs.pydantic.dev/latest/concepts/models/
"""


from pydantic import BaseModel, Field
from typing import Optional, List, Literal



class BaseAlgorithmParams(BaseModel):
    """Base parameters common to all ML models.

    This class provides common configuration parameters that are relevant
    across different algorithm types.

    Attributes:
        random_state: Seed for random number generation (for reproducibility)
        verbose: Flag to control logging verbosity
    """

    random_state: Optional[int] = Field(default=None, description="Seed for random number generation (for reproducibility)")
    verbose: bool = Field(default=False, description="Flag to control logging verbosity")

# its mostly for type checkers
# now they will warn us if we try to assign invalid string
LossType = Literal["mse", "mae"]

# the Optional type tells us that the regularization is optional and can be equal None
RegType = Optional[Literal["l1", "l2", "elasticnet"]]



class SupervisedAlgorithmsParams(BaseAlgorithmParams):
    """Base parameters for all supervised algorithms, we need to think what are the most common parameters

    Attributes:
       test_size: Proportion of the dataset to include in the test split
       validation_size: Proportion of the TRAINING DATA to use as validation
       shuffle: Whether to shuffle the dataset before spliting
       stratify: Whether to use stratify the datasets based on the target values (y)
    """

    test_size: float = Field(
        default=0.2, description="Proportion of the dataset to include in the test split"
    )
    validation_size: Optional[float] = Field(
        default=None,
        gt=0,
        lt=1,
        description="Proportion of the TRAINING DATA to use as validation",
    )
    shuffle: bool = Field(
        default=True, description="Whether to shuffle the dataset before splitting"
    )

    stratify: bool = Field(
        default=False,
        description="Whether to use stratify the datasets based on the target values (y)",
    )


class UnsupervisedAlgorithmsParams(BaseAlgorithmParams):
    """Base parameters for all unsupervised algorithms

    Attributes:
        n_init: Number of times the algorithm will be run
        max_iter: Maximum number of iterations for the algorithm
    """

    n_init: int = Field(
        default=10,
        description="Number of times the algorithm will be run with different centroid seeds",)
    
    max_iter: int = Field(
        default=300,
        description="Maximum number of iterations for the algorithm",)


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

    learning_rate: float = Field(
        default=0.01, description="Step size for gradient optimization"
    )
    epochs: int = Field(
        default=100, description="Number of full passes through the training dataset"
    )
    batch_size: Optional[int] = Field(
        default=None, description="Number of samples per gradient update, None means full batch"
    )

    # -- Regularization Configuration --
    reg_type: RegType = Field(
        default=None,
        description="Type of regularization ('l1', 'l2', 'elasticnet') or None",
    )
    reg_strength: float = Field(
        default=0.01,
        gt=0,
        description="Strength (lambda/alpha) of the regularization. Must be > 0 to have an effect",
    ) 

    mixing_ratio: float = Field(
        default=0.5,
        description="Mixing parameter for ElasticNet regularization. Must be 0 <= mixing_ratio <= 1.",
    )

