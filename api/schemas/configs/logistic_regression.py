from typing import Optional

from schemas.configs.algorithm_configs import GradientBasedParams


# --- ADDED and UPDATED Logistic Regression Params (Inherits GradientBasedParams) ---
class LogisticRegressionParams(GradientBasedParams):
    """
    This model parameter configuration defines the types and possible default arguments
    used by the Logistic Regression model. Inherits GradientBasedParams.

    Attributes:
        threshold: Decision threshold for converting probabilities to class labels (default: 0.5).
        regularization: Type of regularization to apply ("l1", "l2", or None).
        lambda_: Regularization strength (must be non-negative).
    """
    threshold: float = 0.5
    regularization: Optional[str] = None  # "l1", "l2", or None
    lambda_: float = 0.01       # Regularization strength

    def __post_init__(self):
        """Validate parameters after initialization."""
        super().__post_init__()  # Call base class __post_init__ (includes GD validation)

        if not (0.0 < self.threshold < 1.0):
            raise ValueError("threshold must be a value between 0 and 1 (exclusive)")

        # --- Regularization validation ---
        if self.regularization not in (None, "l1", "l2"):
            raise ValueError("regularization must be one of: None, 'l1', or 'l2'")

        if self.lambda_ < 0.0:
            raise ValueError("lambda_ must be a non-negative float")
