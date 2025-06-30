from typing import Literal
from pydantic import Field

from schemas.configs.algorithm_configs import GradientBasedParams

ActivationType = Literal["relu", "tanh", "sigmoid", "gelu", "leaky_relu"]
OptimizerType = Literal["sgd", "adam", "rmsprop"]
LossType = Literal["mse", "mae", "cross_entropy"]


class MLPParams(GradientBasedParams):
    """
    Hyperparameters for a Multi-Layer Perceptron (MLP) model.
    """

    input_size: int = Field(..., gt=0, description="Number of input features")
    hidden_layers: list[int] = Field(
        default_factory=lambda: [64, 64],
        description="Sizes of hidden layers, e.g., [128, 64, 32]",
    )
    output_size: int = Field(..., gt=0, description="Number of output units or classes")
    activation: ActivationType = Field(
        default="relu",
        description="Activation function between layers",
    )
    optimizer: OptimizerType = Field(
        default="adam", description="Optimization algorithm"
    )
    loss: LossType = Field(
        default="mse", description="Loss function to optimize"
    )
    dropout: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Dropout rate for regularization (0 <= dropout <= 1)"
    )
