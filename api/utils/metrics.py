"""
This module provides functions for calculating common evaluation metrics
for machine learning models.
"""

import numpy as np

from utils.losses import MeanAbsoluteError, MeanSquaredError


# Optional: Some metrics are already implemented in the loss classes
# example: MSE/MAE are both losses and metrics


# -- regression metrics --


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) by reusing the Mean Squared Error class.

    MSE = mean((y_pred - y_true)^2)

    Args:
        y_true: Ground truth target values, shape (n_samples,).
        y_pred: Predicted values, shape (n_samples,).

    Returns:
        Mean Squared Error value.
    """

    mse_loss_calculator = MeanSquaredError()

    return mse_loss_calculator(y_true, y_pred)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    class doc string to be filled
    """

    mae_loss_calculator = MeanAbsoluteError()

    return mae_loss_calculator(y_true, y_pred)
