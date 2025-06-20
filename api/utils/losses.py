"""
This module provides implementations of common loss functions used in machine learning,
particularly for gradient-based optimization.

Each loss function is implemented as a class with:
- A __call__ method to compute the scalar loss value.
- A gradient method to compute the gradient of the loss with respect to the predictions (y_pred).
"""

from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    """
    Abstract Base Class for loss functions.
    """

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the loss value.
        """
        pass

    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the loss function with respect to y_pred.
        """
        pass


class MeanSquaredError(LossFunction):
    """
    Mean Squared Error (MSE) loss.
    Loss = mean((y_pred - y_true)^2)
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        if y_true.ndim != 1:
            raise ValueError(f"Expected 1D arrays, got shape {y_true.shape}")
        return np.mean(np.square(y_pred - y_true)).item()

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        if y_true.ndim != 1:
            raise ValueError(f"Expected 1D arrays, got shape {y_true.shape}")
        n_samples = y_true.shape[0]
        if n_samples == 0:
            return np.array([])
        return (2.0 / n_samples) * (y_pred - y_true)


class MeanAbsoluteError(LossFunction):
    """
    Mean Absolute Error (MAE) loss.
    Loss = mean(|y_pred - y_true|)
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        return np.mean(np.abs(y_pred - y_true)).item()

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        grad = np.sign(y_pred - y_true)
        return grad / len(y_true)


class BinaryCrossEntropy(LossFunction):
    """
    Binary Cross-Entropy loss for binary classification tasks.
    BCE = -mean(y * log(y_pred) + (1 - y) * log(1 - y_pred))
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        if y_true.ndim != 1:
            raise ValueError(f"Expected 1D arrays, got shape {y_true.shape}")
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true must contain only binary labels (0 or 1).")
        if not np.all((y_pred >= 0) & (y_pred <= 1)):
            raise ValueError("y_pred must be in range [0, 1].")

        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        if y_true.ndim != 1:
            raise ValueError(f"Expected 1D arrays, got shape {y_true.shape}")
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true must contain only binary labels (0 or 1).")
        if not np.all((y_pred >= 0) & (y_pred <= 1)):
            raise ValueError("y_pred must be in range [0, 1].")

        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return (y_pred - y_true) / (y_pred * (1 - y_pred))


class LogLoss:
    """
    Utility class for computing log loss (used for binary and multiclass classification).
    """

    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        y_true = y_true.astype(float)

        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        y_true = y_true.astype(float)
        return -(y_true / y_pred) + ((1 - y_true) / (1 - y_pred))
