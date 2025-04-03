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

# --- Define the type for regularization ---
RegType = Literal['l1', 'l2', 'elasticnet']

@dataclass
class LogisticRegressionParams(SupervisedAlgorithmsParams):
    """
    This model parameter configuration defines the types and possible default arguments
    used by the Logistic Regression model.

    Attributes:
        learning_rate: Step size for gradient optimization (default: 0.01).
        epochs: Number of full passes through the training dataset (default: 100).
        batch_size: Number of samples per gradient update, None means the entire dataset (full batch) (default: None).
        threshold: Decision threshold for converting probabilities to class labels (default: 0.5).

        reg_type: Type of regularization ('l1', 'l2', 'elasticnet') or None (default: None).
        reg_strength: Strength (lambda/alpha) of the regularization. Must be > 0 to have an effect (default: 0.01).
        mixing_ratio: Mixing parameter for ElasticNet regularization. Must be 0 <= mixing_ratio <= 1.
                      A value of 0.0 corresponds to L2 only, and 1.0 to L1 only.
                      Used only when reg_type='elasticnet' (default: 0.5).
        # Note: The 'loss' parameter is typically constant for logistic regression
        # (e.g., log loss / binary cross-entropy) and might not be configurable
        # in the same way as in linear regression. For clarity, it's omitted here,
        # assuming the model implementation will use the appropriate loss function.
    """

    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: Optional[int] = None
    threshold: float = 0.5

    # -- Regularization Configuration --
    reg_type: Optional[RegType] = None
    reg_strength: float = 0.01  # Corrected typo from 'strenght'
    mixing_ratio: float = 0.5   # Default mixing ratio for ElasticNet

    def __post_init__(self):
        """Validate parameters after initialization."""
        super().__post_init__() # Call base class __post_init__

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer or None")
        if not (0.0 < self.threshold < 1.0):
             raise ValueError("threshold must be a value between 0 and 1 (exclusive)")

        # Use the corrected name reg_strength
        if self.reg_type is not None and self.reg_strength <= 0:
            raise ValueError("reg_strength must be greater than 0 when regularization is used")

        if self.reg_type == "elasticnet":
            if not (0.0 <= self.mixing_ratio <= 1.0):
                raise ValueError("mixing_ratio must be between 0 and 1 for ElasticNet")
        # Check if 'mixing_ratio' is set inappropriately
        elif self.reg_type is not None and self.mixing_ratio != 0.5: # Check if the value is non-default
             # Use sys.stderr for warnings
             print(f"Warning: parameter 'mixing_ratio' ({self.mixing_ratio}) is set, "
                   f"but it only has an effect when reg_type='elasticnet'. Current type: {self.reg_type}",
                   file=sys.stderr)
