from dataclasses import dataclass, field
from typing import Literal, Optional, Union

# Types for parameters
ActivationType = Literal["relu", "tanh", "sigmoid", "softmax"]
OptimizerType = Literal["sgd", "adam", "rmsprop", "adagrad", "adadelta", "adamw"]
LossType = Literal["mse", "mae", "cross_entropy"]

@dataclass
class MLPParams(SupervisedAlgorithmsParams):
    """
    Configuration parameters for the MLP (Multilayer Perceptron) model.

    Attributes:
        hidden_layers: List of neurons per hidden layer (e.g., [128, 64, 32]).
        dropout: Fraction of neurons to drop during training (0.0 < dropout <= 1.0).
        l2_reg: L2 regularization coefficient (None means no regularization).
        learning_rate: Global or layer-wise learning rate (float or list of floats).
        activation: Activation function used in hidden layers.
        optimizer: Optimization algorithm.
    """
    hidden_layers: list[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.2  # Must be > 0.0 for regularization to have effect
    l2_reg: Optional[float] = None
    learning_rate: Union[float, list[float]] = 0.001
    activation: ActivationType = "relu"
    optimizer: OptimizerType = "adam"

    def __post_init__(self):
        super().__post_init__()
        if any(h <= 0 for h in self.hidden_layers):
            raise ValueError("All hidden layers must have a positive number of neurons.")
        if not (0.0 < self.dropout <= 1.0):
            raise ValueError("Dropout must be in the range (0.0, 1.0].")
        if self.l2_reg is not None and self.l2_reg < 0:
            raise ValueError("L2 regularization coefficient must be >= 0.")
        if isinstance(self.learning_rate, list):
            if len(self.learning_rate) != len(self.hidden_layers):
                raise ValueError("Length of learning_rate list must match number of hidden layers.")
            if any(lr <= 0 for lr in self.learning_rate):
                raise ValueError("All learning rates must be positive.")
        elif self.learning_rate <= 0:
            raise ValueError("Learning rate must be a positive value.")
