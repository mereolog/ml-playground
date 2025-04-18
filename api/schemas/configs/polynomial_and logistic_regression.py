import sys
from dataclasses import dataclass
from typing import Optional, Literal

# --- Base class for all supervised algorithm parameters ---
@dataclass
class SupervisedAlgorithmsParams:
    """
    Base class for supervised algorithm parameters.
    """
    def __post_init__(self):
        pass


# --- Define common types ---
RegType = Literal['l1', 'l2', 'elasticnet']
LossType = Literal['mse', 'mae']


@dataclass
class RegularizedLinearModelParams(SupervisedAlgorithmsParams):
    """
    Generalized class for models using linear learning and regularization.

    Attributes:
        learning_rate: Step size for gradient descent (default: 0.01).
        epochs: Number of training iterations (default: 100).
        batch_size: Number of samples per update. If None, full batch is used (default: None).
        reg_type: Type of regularization to apply: 'l1', 'l2', or 'elasticnet' (default: None).
        reg_strength: Regularization strength (default: 0.01). Must be > 0 when reg_type is set.
        mixing_ratio: Used for ElasticNet; 0.0 is L2, 1.0 is L1. (default: 0.5).
    """
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: Optional[int] = None

    reg_type: Optional[RegType] = None
    reg_strength: float = 0.01
    mixing_ratio: float = 0.5

    def __post_init__(self):
        super().__post_init__()

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer or None")

        if self.reg_type is not None and self.reg_strength <= 0:
            raise ValueError("reg_strength must be greater than 0 when regularization is used")

        if self.reg_type == "elasticnet":
            if not (0.0 <= self.mixing_ratio <= 1.0):
                raise ValueError("mixing_ratio must be between 0 and 1 for ElasticNet")
        elif self.reg_type is not None and self.mixing_ratio != 0.5:
            print(f"Warning: parameter 'mixing_ratio' ({self.mixing_ratio}) is set, "
                  f"but it only has an effect when reg_type='elasticnet'. Current type: {self.reg_type}",
                  file=sys.stderr)


@dataclass
class LogisticRegressionParams(RegularizedLinearModelParams):
    """
    Parameters specific to logistic regression models.

    Attributes:
        threshold: Decision threshold for converting predicted probabilities to class labels (default: 0.5).
    """
    threshold: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if not (0.0 < self.threshold < 1.0):
            raise ValueError("threshold must be a value between 0 and 1 (exclusive)")


@dataclass
class PolynomialRegressionParams(RegularizedLinearModelParams):
    """
    Parameters specific to polynomial regression models.

    Attributes:
        degree: Degree of the polynomial features (default: 2). Must be >= 1.
        include_bias: Whether to include a bias term in the polynomial features (default: True).
        loss: Loss function to use: 'mse' or 'mae' (default: 'mse').
    """
    degree: int = 2
    include_bias: bool = True
    loss: LossType = "mse"

    def __post_init__(self):
        super().__post_init__()

        if self.degree < 1:
            raise ValueError("degree must be an integer >= 1")

        allowed_losses = LossType.__args__
        if self.loss not in allowed_losses:
            raise ValueError(f"Invalid loss function for regression: {self.loss}. Allowed: {allowed_losses}.")
