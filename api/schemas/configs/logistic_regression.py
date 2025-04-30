from dataclasses import dataclass

from schemas.configs.algorithms_configs import GradientBasedParams


# --- ADDED and UPDATED Logistic Regression Params (Inherits GradientBasedParams) ---
@dataclass
class LogisticRegressionParams(GradientBasedParams):
    """
    This model parameter configuration defines the types and possible default arguments
    used by the Logistic Regression model. Inherits GradientBasedParams.

    Attributes:
        threshold: Decision threshold for converting probabilities to class labels (default: 0.5).
        # Note: Loss is typically binary cross-entropy/log loss for logistic regression
        # and is usually not configurable in the same way as regression loss.
        # It's not included here, assuming the model implementation handles it.
    """
    # Parameters learning_rate, epochs, batch_size, reg_type, reg_strength, mixing_ratio
    # are inherited from GradientBasedParams

    threshold: float = 0.5

    def __post_init__(self):
        """Validate parameters after initialization."""
        super().__post_init__() # Call base class __post_init__ (includes GD validation)

        # --- Threshold validation (specific to Logistic Regression) ---
        if not (0.0 < self.threshold < 1.0):
            raise ValueError("threshold must be a value between 0 and 1 (exclusive)")
