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



def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)

def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the R² (coefficient of determination) regression score.
    
    Args:
        y_true: Ground truth target values.
        y_pred: Estimated target values.
        
    Returns:
        R² score.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)