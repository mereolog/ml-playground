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

        Args:
            y_true: Ground truth target values.
            y_pred: Predicted values.

        Returns:
            Float as scalar loss value.
        """
        pass

    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the loss function with respect to y_pred.

        Args:
            y_true: Ground truth target values.
            y_pred: Predicted values.

        Returns:
            Gradient vector, shape (n_samples,).
        """
        pass


class MeanSquaredError(LossFunction):
    """
    Mean Squared Error (MSE) loss.

    Loss = mean((y_pred - y_true)^2)
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculates the mean squared error."""
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )
        if y_true.ndim != 1:  # Just for simplicity lets assume the values are 1D
            raise ValueError(f"Expected 1D arrays, got shape {y_true.shape}")
        return np.mean(
            np.square(y_pred - y_true)
        ).item()  # .item() converts element of numpy np.ndarray to standard python scalar

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculates the gradient of MSE w.r.t. y_pred."""
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            )
        if y_true.ndim != 1:
            raise ValueError(f"Expected 1D arrays, got shape {y_true.shape}")
        n_samples = y_true.shape[0]
        if n_samples == 0:
            return np.array([])  # if the input is empty lets just return empty list
        # Gradient is (2 / n) * (y_pred - y_true)
        # this calculates derivative (gradient) with respect to the predictions
        return (2.0 / n_samples) * (y_pred - y_true)


class MeanAbsoluteError(LossFunction):
    """
    replace this with valid doc string
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # your implementation
        pass

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class BinaryCrossEntropy(LossFunction):
    """
    same here
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # implementation
        pass

class LogLoss:
    """Implementation of the logarithmic Loss (Log Loss) function.
    Log loss is used to evaluate the performance of a classification model where
    the output is a probability value between 0 and 1.

    Formula:
     LogLoss = -1/N * Σ (y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

    Attributes:
        None
    """
    @staticmethod
    def compute(y_true: np.ndarray, y_pred:np.ndarray) -> float:
        """
        Compute the Log Loss value

         Args:
            y_true: Ground truth labels, shape (n_samples,).
                    Labels should be binary (0 or 1) or one-hot encoded for multi-class problems.
            y_pred: Predicted probabilities, shape (n_samples,) for binary
                    or (n_samples, n_classes) for multi-class.

        Returns:
            Log Loss value (scalar).
        """
        # Ensure predictions are clipped to avoid log(0) or log(1) errors
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # For binary classification
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            log_loss = -np.mean(
                y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
            )
        else:  # For multi-class classification
            log_loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

        return log_loss

    @staticmethod
    def gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the Log Loss with respect to the predictions.

        Args:
            y_true: Ground truth labels, shape (n_samples,).
                    Labels should be binary (0 or 1) or one-hot encoded for multi-class problems.
            y_pred: Predicted probabilities, shape (n_samples,) for binary
                    or (n_samples, n_classes) for multi-class.

        Returns:
            Gradient of Log Loss with respect to predictions, shape as y_pred.
        """
        # Ensure predictions are clipped to avoid division by zero
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Gradient computation
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)




