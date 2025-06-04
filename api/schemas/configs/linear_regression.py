from pydantic import Field

from schemas.configs.algorithms_configs import GradientBasedParams, LossType


class LinearRegressionParams(GradientBasedParams):
    """This model parameters config defines types and possible default arguments used by LinearRegression model
    Inherits GradientBasedParams for optimization and regularization settings.

    Attributes:
        loss: Loss function to use ('mse', 'mae') (default: 'mse')
        # Removed the redundant 'regularization' attribute from the original file.
    """

    # -- loss function config --
    loss: LossType = Field(
        default="mse",
        description="Loss function to use for regression. Options: 'mse' (Mean Squared Error), 'mae' (Mean Absolute Error).",
    )