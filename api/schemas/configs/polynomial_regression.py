from dataclasses import dataclass

from schemas.configs.algorithm_configs import GradientBasedParams, LossType


# --- ADDED and UPDATED Polynomial Regression Params (Inherits GradientBasedParams) ---

class PolynomialRegressionParams(GradientBasedParams):
    """
    This model parameter configuration defines the types and possible default arguments
    used by the Polynomial Regression model. Inherits GradientBasedParams.

    Polynomial regression typically transforms input features into polynomial features
    and then applies linear regression. Therefore, many parameters are similar
    to linear regression, but with the addition of the polynomial degree.

    Attributes:
        degree: The degree of the polynomial features to generate (default: 2).
        include_bias: Whether to include a bias column (column of ones) during feature transformation (default: True).
                      This is often handled by the underlying linear regression model itself.
        loss: Loss function used by the underlying linear regression ('mse', 'mae') (default: 'mse').
              (Moved from original snippet, belongs here as it's regression specific)
    """
    # Parameters learning_rate, epochs, batch_size, reg_type, reg_strength, mixing_ratio
    # are inherited from GradientBasedParams

    degree: int = 2
    include_bias: bool = True
    loss: LossType = "mse" 

    def __post_init__(self):
        """Validate parameters after initialization."""
        super().__post_init__() 

        # --- Specific validation for Polynomial Regression ---
        if self.degree < 1:
            raise ValueError("degree must be an integer >= 1")

        # --- Loss validation (specific to regression models) ---
        allowed_losses = LossType.__args__
        if self.loss not in allowed_losses:
            raise ValueError(f"Invalid loss function for regression: {self.loss}. Allowed: {allowed_losses}.")
