from typing import Callable

import torch
from torch import nn, optim

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

    @staticmethod
    def _get_activation(name: str) -> Callable:
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "sigmoid":
            return nn.Sigmoid()
        elif name == "gelu":
            return nn.GELU()
        elif name == "leaky_relu":
            return nn.LeakyReLU()
        raise ValueError(f"Unsupported activation: {name}")


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


def _get_loss_function(name: str) -> Callable:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    elif name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss function: {name}")


def _get_optimizer(name: str, parameters, lr: float) -> optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return optim.Adam(parameters, lr=lr)
    elif name == "sgd":
        return optim.SGD(parameters, lr=lr)
    elif name == "rmsprop":
        return optim.RMSprop(parameters, lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}")
