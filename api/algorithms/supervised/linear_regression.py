# algorithms/supervised/linear_regression.py
"""
This module contains implementation of the Linear Regression algorithm.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from algorithms.base.supervised import SupervisedAlgorithm
from schemas.configs.algorithms_configs import LinearRegressionParams
from utils.losses import LossFunction, MeanAbsoluteError, MeanSquaredError
from utils.metrics import mean_absolute_error, mean_squared_error


class LinearRegression(SupervisedAlgorithm[LinearRegressionParams]):
    """
    Linear Regression implementation using gradient descent.

    This class implements linear regression using gradient descent optimization
    to find the best-fitting line for the input data.
    """

    def __init__(self, params: Optional[LinearRegressionParams] = None):
        """
        Initialize the linear regression model.

        Args:
            params: Configuration parameters for the model. Uses defaults if None.
        """
        super().__init__()

        # -- use provided parameters or fallback to defaults
        self._params = params if params is not None else LinearRegressionParams()

        self._loss_fn: LossFunction

        # -- instantiate the correct loss function from param config
        loss_name = self._params.loss
        match loss_name:
            case "mse":
                self._loss_fn = MeanSquaredError()
            case "mae":
                self._loss_fn = MeanAbsoluteError()
            # other loss functions go will go here
            case _:  # sanity check for safety, though Literal should prevent this
                raise ValueError(f"Unknown loss function name in params: {loss_name}")
        # -- end

        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.loss_history: List[float] = []

        # -- set up logger based on verbose setting
        self.logger = logging.getLogger(__name__)
        if self.params.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

    @property
    def params(self) -> LinearRegressionParams:
        """Returns the parameters for this algorithm instance."""
        return self._params

    def _initialize_parameters(self, n_features: int) -> None:
        """
        Initialize model weights and bias.

        Args:
            n_features: Number of input features.
        """
        if self.params.random_state is not None:
            np.random.seed(self.params.random_state)

        # Small random weights, zero bias is a common starting point
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.logger.info("Model parameters initialized.")

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Make raw predictions using the linear model y = Xw + b.
        Assumes weights and bias are initialized.

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            Predicted values, shape (n_samples,).

        Raises:
            ValueError: If weights or bias are not initialized.
        """
        if self.weights is None or self.bias is None:
            # This should ideally not be reached if called after _initialize_parameters
            raise ValueError("Model parameters (weights/bias) are not initialized.")
        return np.dot(X, self.weights) + self.bias

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the loss, including the chosen loss function and regularization penalties.

        Args:
            X: Input features, shape (n_samples, n_features).
            y: Target values, shape (n_samples,).

        Returns:
            Scalar loss value.
        """
        n_samples = X.shape[0]
        if n_samples == 0:
            return 0.0
        if self.weights is None:  # Should be initialized before calling this
            raise ValueError("Cannot compute loss, weights are not initialized.")

        predictions = self._predict_raw(X)
        # Calculate loss from the chosen loss function
        base_loss = self._loss_fn(y, predictions)

        # add regularization penalty (applied only to weights)
        # here we need to implement regularization logic
        reg_strength = self.params.reg_strenght
        reg_type = self.params.reg_type
        # loss = base_loss + reg_penealty or something like that

        return base_loss

    def _compute_gradients(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weights and bias, including regularization terms.

        Args:
            X: Input features, shape (n_samples, n_features).
            y: Target values, shape (n_samples,).

        Returns:
            Tuple of (weight gradients, bias gradient).
        """
        n_samples = X.shape[0]
        if n_samples == 0:
            # Return zero gradients for empty batch
            if self.weights is not None:
                return np.zeros_like(self.weights), 0.0
            else:
                raise ValueError("Cannot compute gradients, weights not initialized.")
        if self.weights is None:  # Should be initialized
            raise ValueError("Cannot compute gradients, weights are not initialized.")

        predictions = self._predict_raw(X)

        # Gradient of the base loss function w.r.t. predictions
        loss_gradient_wrt_pred = self._loss_fn.gradient(y, predictions)

        # Gradients from the loss function part (using chain rule)
        # dw = dLoss/dy_pred * d(y_pred)/dw = loss_gradient_wrt_pred * X.T
        # db = dLoss/dy_pred * d(y_pred)/db = loss_gradient_wrt_pred * 1
        dw = np.dot(X.T, loss_gradient_wrt_pred)
        db = np.sum(loss_gradient_wrt_pred)

        # we should add regularization here, no?

        return dw, db

    def _update_parameters(self, dw: np.ndarray, db: float) -> None:
        """
        Update model parameters using calculated gradients and learning rate.

        Args:
            dw: Weight gradients.
            db: Bias gradient.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Cannot update parameters, model not initialized.")

        self.weights -= self.params.learning_rate * dw
        self.bias -= self.params.learning_rate * db

    def _get_batch_indices(self, n_samples: int, batch_size: int) -> np.ndarray:
        """
        Get random batch indices. Handles random state.

        Args:
            n_samples: Total number of samples.
            batch_size: Size of the batch.

        Returns:
            Array of random indices.
        """
        # Use a local RandomState based on the main random_state if provided
        # to avoid interfering with global numpy random state if batches are drawn sequentially
        local_rng = np.random.RandomState(self.params.random_state)
        indices = local_rng.permutation(n_samples)
        # If random_state changes, subsequent calls might yield different permutations
        # For perfect reproducibility across calls, might need more complex state handling
        return indices[:batch_size]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Fit the linear regression model to the training data using gradient descent.

        Args:
            X: Training data features, shape (n_samples, n_features).
            y: Target values, shape (n_samples,).

        Returns:
            Self reference for method chaining.
        """
        n_samples, n_features = X.shape
        if n_samples == 0:
            self.logger.warning("Fitting skipped: Received empty training data.")
            return self

        self._initialize_parameters(n_features)

        actual_batch_size = (
            n_samples if self.params.batch_size is None else self.params.batch_size
        )

        for epoch in range(self.params.epochs):

            if self.params.batch_size is None:
                # full batch gradient descent
                dw, db = self._compute_gradients(X, y)
                self._update_parameters(dw, db)
                # Loss computation for full batch is done once below
            else:
                # gradient descent on randomly shuffled batches
                for i in range(0, n_samples, actual_batch_size):
                    batch_indices = self._get_batch_indices(
                        n_samples, actual_batch_size
                    )  # Or slice sequentially: indices[i:i+batch_size]
                    X_batch = X[batch_indices]
                    y_batch = y[batch_indices]

                    if X_batch.shape[0] == 0:
                        continue  # Skip empty batches

                    dw, db = self._compute_gradients(X_batch, y_batch)
                    self._update_parameters(dw, db)

            # Compute and store loss for the epoch (using the full dataset for consistency)
            current_loss = self._compute_loss(X, y)
            self.loss_history.append(current_loss)

            # Log progress if verbose
            if self.params.verbose and (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.params.epochs}, Loss: {current_loss:.6f}"
                )

        self.logger.info(f"Training finished after {self.params.epochs} epochs.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the trained linear model.

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            Predicted values, shape (n_samples,).

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained. Call fit() before predict().")
        return self._predict_raw(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate the coefficient of determination (R^2) for the model.

        Args:
            X: Test features, shape (n_samples, n_features).
            y: True target values, shape (n_samples,).

        Returns:
            Dictionary containing scores

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained. Call fit() before score().")

        y_pred = self.predict(X)

        scores = {
            "mse": mean_squared_error(y, y_pred),
            "mae": mean_absolute_error(y, y_pred),
            # other scores that need implementing
        }

        return scores

    def get_coefficients(self) -> Dict[str, Any]:
        """
        Get the learned model coefficients (weights and bias).

        Returns:
            Dictionary containing the learned 'weights' and 'bias'.

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.weights is None or self.bias is None:
            raise ValueError(
                "Model has not been trained. Call fit() before get_coefficients()."
            )
        return {"weights": self.weights.copy(), "bias": self.bias}

    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history (e.g., loss per epoch).

        Returns:
            Dictionary containing the training history data.
        """

        training_history = {"training_loss": self.loss_history}
        return training_history
