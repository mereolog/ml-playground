import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable
from schemas.configs.MLP_config import MLPParams


class MLP(nn.Module):
    def __init__(self, config: MLPParams):
        super().__init__()
        self.config = config

        layers = []
        in_features = config.input_size

        for hidden_size in config.hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(self._get_activation(config.activation))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, config.output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _get_activation(self, name: str) -> Callable:
        match name:
            case "relu": return nn.ReLU()
            case "tanh": return nn.Tanh()
            case "sigmoid": return nn.Sigmoid()
            case "gelu": return nn.GELU()
            case "leaky_relu": return nn.LeakyReLU()
            case _: raise ValueError(f"Unsupported activation: {name}")


def train_mlp(model: MLP, X: torch.Tensor, y: torch.Tensor, config: MLPParams) -> None:
    criterion = _get_loss_function(config.loss)
    optimizer = _get_optimizer(config.optimizer, model.parameters(), config.learning_rate)

    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size or len(X), shuffle=True)

    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if config.verbose:
            print(f"Epoch {epoch + 1}/{config.epochs}, Loss: {epoch_loss:.4f}")


def _get_loss_function(name: str):
    match name:
        case "mse": return nn.MSELoss()
        case "mae": return nn.L1Loss()
        case "cross_entropy": return nn.CrossEntropyLoss()
        case _: raise ValueError(f"Unsupported loss function: {name}")


def _get_optimizer(name: str, parameters, lr: float):
    match name:
        case "adam": return optim.Adam(parameters, lr=lr)
        case "sgd": return optim.SGD(parameters, lr=lr)
        case "rmsprop": return optim.RMSprop(parameters, lr=lr)
        case _: raise ValueError(f"Unsupported optimizer: {name}")
