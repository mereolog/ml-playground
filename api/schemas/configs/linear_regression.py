from dataclasses import dataclass

from schemas.configs.algorithms_configs import GradientBasedParams, LossType


# --- UPDATED Linear Regression Params (Inherits GradientBasedParams) ---
@dataclass
class LinearRegressionParams(GradientBasedParams):
    """This model parameters config defines types and possible default arguments used by LinearRegression model
    Inherits GradientBasedParams for optimization and regularization settings.

    Attributes:
        loss: Loss function to use ('mse', 'mae') (default: 'mse')
        # Removed the redundant 'regularization' attribute from the original file.
    """
    # Parameters learning_rate, epochs, batch_size, reg_type, reg_strength, mixing_ratio
    # are inherited from GradientBasedParams

    # -- loss function config --
    loss: LossType = "mse"

    def __post_init__(self):
        super().__post_init__() # Call parent __post_init__ (includes GD, Supervised, Base validation)

        # --- Loss validation (specific to regression models) ---
        allowed_losses = LossType.__args__
        if self.loss not in allowed_losses:
            raise ValueError(f"Invalid loss function for regression: {self.loss}. Allowed: {allowed_losses}.")



