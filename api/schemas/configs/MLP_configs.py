from typing import Literal
from pydantic import Field

from schemas.configs.algorithm_configs import GradientBasedParams

ActivationType = Literal["relu", "tanh", "sigmoid", "gelu", "leaky_relu"]


class MLPParams(GradientBasedParams):
    """
    Hyperparameters for a Multi-Layer Perceptron (MLP) model.
    Inherits from GradientBasedParams, which already includes training-related settings.

    Attributes:
        input_size: Number of input features (required).
        hidden_layers: List specifying the size of each hidden layer (default: [64, 64]).
        output_size: Number of output units (required).
        activation: Activation function used between layers (default: 'relu').
    """

    input_size: int = Field(
        ..., gt=0, description="Number of input features"
    )
    hidden_layers: list[int] = Field(
        default_factory=lambda: [64, 64],
        description="Sizes of hidden layers, e.g., [128, 64, 32]",
    )
    output_size: int = Field(
        ..., gt=0, description="Number of output units or classes"
    )
    activation: ActivationType = Field(
        default="relu",
        description="Activation function to use between layers (e.g., 'relu', 'tanh')",
    )
