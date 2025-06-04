from pydantic import Field
from schemas.configs.algorithm_configs import GradientBasedParams


class LogisticRegressionParams(GradientBasedParams):
    """
    Pydantic model for Logistic Regression algorithm parameters.
    This model parameter configuration defines the types and possible default arguments
    used by the Logistic Regression model. Inherits GradientBasedParams.

    Attributes:
        threshold: Decision threshold for converting probabilities to class labels (default: 0.5). Must be between 0 and 1 (exclusive).
        # Parameters like learning_rate, epochs, batch_size, reg_type, reg_strength, mixing_ratio
        # are inherited from GradientBasedParams.
        # Loss is typically binary cross-entropy/log loss for logistic regression
        # and is usually not configurable in the same way as regression loss.
    """

    threshold: float = Field(
        default=0.5,
        description="Decision threshold for converting probabilities to class labels.",
        gt=0.0,
        lt=1.0,
    )
