from dataclasses import dataclass, field
from typing import Literal, Optional

# Types for parameters
ActivationType = Literal["relu", "tanh", "sigmoid"]
OptimizerType = Literal["sgd", "adam", "rmsprop"]
LossType = Literal["mse", "mae", "cross_entropy"]

@dataclass
class MLPParams(SupervisedAlgorithmsParams):
    """Configuration parameters for the MLP (Multilayer Perceptron) model.

    Attributes:
        hidden_layers: List specifying the number of neurons in each hidden layer.
        activation: Activation function for hidden layers ('relu', 'tanh', 'sigmoid').
        optimizer: Optimization algorithm ('sgd', 'adam', 'rmsprop').
        learning_rate: Learning rate.
        epochs: Number of training epochs.
        batch_size: Batch size (None means full batch).
        loss: Loss function ('mse', 'mae', 'cross_entropy').
        dropout: Percentage of neurons dropped out during regularization (0-1).
        l2_reg: L2 regularization coefficient (default None - no regularization).
    """
    hidden_layers: list[int] = field(default_factory=lambda: [64, 32])
    activation: ActivationType = "relu"
    optimizer: OptimizerType = "adam"
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: Optional[int] = None
    loss: LossType = "mse"
    dropout: float = 0.0
    l2_reg: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        if any(h <= 0 for h in self.hidden_layers):
            raise ValueError("All hidden layers must have a positive number of neurons.")
        if not (0 <= self.dropout < 1):
            raise ValueError("Dropout must be a value in the range [0, 1).")
        if self.l2_reg is not None and self.l2_reg < 0:
            raise ValueError("L2 regularization coefficient must be >= 0.")