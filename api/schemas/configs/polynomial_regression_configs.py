from pydantic import Field

from schemas.configs.algorithm_configs import GradientBasedParams, LossType


class PolynomialRegressionParams(GradientBasedParams):
    """
    Pydantic model for Polynomial Regression parameters.
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

    degree: int = Field(
        2,
        ge=1,
        description="The degree of the polynomial features to generate (must be >= 1, default: 2).",
    )
    include_bias: bool = Field(
        True,
        description="Whether to include a bias column (column of ones) during feature transformation (default: True).",
    )
    loss: LossType = Field(
        "mse",
        description="Loss function used by the underlying linear regression ('mse', 'mae') (default: 'mse').",
    )
