import sys  
from dataclasses import dataclass
from typing import Optional, Literal

# --- Assumed base configuration class ---
@dataclass
class SupervisedAlgorithmsParams:
    """Base class for supervised algorithm parameters."""
    def __post_init__(self):
        """Perform basic validation if needed."""
        pass # Base validation, if any

# --- Define helper types ---
# Using Literal for simplicity, could be replaced with an Enum
LossType = Literal['mse', 'mae']
RegType = Literal['l1', 'l2', 'elasticnet']

@dataclass
class PolynomialRegressionParams(SupervisedAlgorithmsParams):
    """
    This model parameter configuration defines the types and possible default arguments
    used by the Polynomial Regression model.

    Polynomial regression typically transforms input features into polynomial features
    and then applies linear regression. Therefore, many parameters are similar
    to linear regression, but with the addition of the polynomial degree.

    Attributes:
        degree: The degree of the polynomial features to generate (default: 2).
        include_bias: Whether to include a bias column (column of ones) during feature transformation (default: True).
                      This is often handled by the underlying linear regression model itself.

        # Parameters inherited or adapted from linear regression,
        # applied to the linear model on the transformed features:
        learning_rate: Step size for gradient optimization (if used) (default: 0.01).
        epochs: Number of epochs for gradient optimization (if used) (default: 100).
        batch_size: Batch size for gradient optimization (if used) (default: None).

        loss: Loss function used by the underlying linear regression ('mse', 'mae') (default: 'mse').

        reg_type: Type of regularization ('l1', 'l2', 'elasticnet') applied to the linear regression (default: None).
        reg_strength: Strength (lambda/alpha) of the regularization (default: 0.01). 
        mixing_ratio: Mixing parameter for ElasticNet (default: 0.5).
    """

    degree: int = 2
    include_bias: bool = True

    # -- Parameters for the underlying linear regression (if the implementation uses them) --
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: Optional[int] = None

    # -- Loss function configuration --
    loss: LossType = "mse" # MSE is standard for regression

    # -- Regularization configuration --
    reg_type: Optional[RegType] = None 
    reg_strength: float = 0.01       

    def __post_init__(self):
        """Validate parameters after initialization."""
        super().__post_init__() 

        if self.degree < 1:
            raise ValueError("degree must be an integer >= 1")

        # --- Validation for underlying linear model parameters ---
        if self.learning_rate <= 0:
             raise ValueError("learning_rate must be greater than 0")
        if self.epochs <= 0:
             raise ValueError("epochs must be a positive integer")
        if self.batch_size is not None and self.batch_size <= 0:
             raise ValueError("batch_size must be a positive integer or None")

        # --- Loss validation ---
        # The Literal type hint already provides some level of check if using static analysis tools,
        # but runtime check is still good.
        allowed_losses = LossType.__args__ 
        if self.loss not in allowed_losses:
             raise ValueError(f"Invalid loss function for regression: {self.loss}. Allowed: {allowed_losses}.")

        # --- Regularization validation (using corrected reg_strength) ---
        if self.reg_type is not None and self.reg_strength <= 0:
            raise ValueError("reg_strength must be greater than 0 when regularization is used")

        if self.reg_type == "elasticnet":
            if not (0.0 <= self.mixing_ratio <= 1.0):
                raise ValueError("mixing_ratio must be between 0 and 1 for ElasticNet")
        elif self.reg_type is not None and self.mixing_ratio != 0.5: 
             print(f"Warning: parameter 'mixing_ratio' ({self.mixing_ratio}) is set, "
                   f"but it only has an effect when reg_type='elasticnet'. Current type: {self.reg_type}",
                   file=sys.stderr)