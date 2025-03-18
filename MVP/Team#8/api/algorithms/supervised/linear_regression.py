# algorithms/supervised/linear_regression.py

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from algorithms.base.supervised import SupervisedAlgorithm
from schemas.configs.algorithms_configs import LinearRegressionParams


class LinearRegression(SupervisedAlgorithm[LinearRegressionParams]):
    """
    Linear Regression implementation using gradient descent.

    This class implements linear regression using gradient descent optimization
    to find the best-fitting line for the input data.
    """

    def __init__(self, params: Optional[LinearRegressionParams] = None):
        """
        Initialize the linear regression model with given parameters.

        Args:
            params: Configuration parameters for the model
        """
        super().__init__()
        self._params = params if params is not None else LinearRegressionParams()

        # Model parameters to be learned
        self.weights = None
        self.bias = None

        # Training history
        self.loss_history = []

        # Setup logging
        self.logger = logging.getLogger(__name__)
        if self.params.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

    @property
    def params(self) -> LinearRegressionParams:
        return self._params

    def _initialize_parameters(self, n_features: int) -> None:
        """
        Initialize model parameters.

        Args:
            n_features: Number of input features
        """
        # Set random seed for reproducibility if specified
        if self.params.random_state is not None:
            np.random.seed(self.params.random_state)

        # Initialize weights and bias
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the mean squared error loss.

        Args:
            X: Input features, shape (n_samples, n_features)
            y: Target values, shape (n_samples,)

        Returns:
            Mean squared error loss
        """
        n_samples = X.shape[0]
        predictions = self._predict_raw(X)

        # Calculate mean squared error
        mse = np.mean((predictions - y) ** 2)

        # Add L2 regularization if specified
        if self.params.regularization is not None:
            l2_reg = self.params.regularization * np.sum(self.weights**2)
            mse += l2_reg

        return mse

    def _compute_gradients(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weights and bias.

        Args:
            X: Input features, shape (n_samples, n_features)
            y: Target values, shape (n_samples,)

        Returns:
            Tuple of (weight gradients, bias gradient)
        """
        n_samples = X.shape[0]
        predictions = self._predict_raw(X)

        # Calculate gradients
        dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
        db = (1 / n_samples) * np.sum(predictions - y)

        # Add L2 regularization gradient if specified
        if self.params.regularization is not None:
            dw += 2 * self.params.regularization * self.weights

        return dw, db

    def _update_parameters(self, dw: np.ndarray, db: float) -> None:
        """
        Update model parameters using calculated gradients.

        Args:
            dw: Weight gradients
            db: Bias gradient
        """
        self.weights -= self.params.learning_rate * dw
        self.bias -= self.params.learning_rate * db

    def _get_batch_indices(self, n_samples: int, batch_size: int) -> np.ndarray:
        """
        Get random batch indices.

        Args:
            n_samples: Total number of samples
            batch_size: Size of the batch

        Returns:
            Array of random indices
        """
        indices = np.random.permutation(n_samples)
        return indices[:batch_size]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Fit the linear regression model to the training data.

        Args:
            X: Training data features, shape (n_samples, n_features)
            y: Target values, shape (n_samples,)

        Returns:
            Self reference for method chaining
        """
        n_samples, n_features = X.shape

        # Initialize model parameters
        self._initialize_parameters(n_features)

        # Clear loss history from previous training
        self.loss_history = []

        # Training loop
        for epoch in range(self.params.epochs):
            if self.params.batch_size is None:
                # Full batch gradient descent
                dw, db = self._compute_gradients(X, y)
                self._update_parameters(dw, db)
            else:
                # Mini-batch gradient descent
                batch_size = min(self.params.batch_size, n_samples)
                batch_indices = self._get_batch_indices(n_samples, batch_size)

                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                dw, db = self._compute_gradients(X_batch, y_batch)
                self._update_parameters(dw, db)

            # Compute and store loss
            current_loss = self._compute_loss(X, y)
            self.loss_history.append(current_loss)

            # Log progress if verbose
            if self.params.verbose and (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.params.epochs}, Loss: {current_loss:.6f}"
                )

        return self

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Make raw predictions using the linear model.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Predicted values, shape (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Predicted values, shape (n_samples,)
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained. Call fit() before predict().")

        return self._predict_raw(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the coefficient of determination (R^2) for the model.

        Args:
            X: Test features, shape (n_samples, n_features)
            y: True target values, shape (n_samples,)

        Returns:
            R^2 score
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained. Call fit() before score().")

        y_pred = self.predict(X)

        # Calculate R^2
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)

        if ss_total == 0:
            return 0  # Avoid division by zero

        return 1 - (ss_residual / ss_total)

    def get_coefficients(self) -> Dict[str, Any]:
        """
        Get the model coefficients.

        Returns:
            Dictionary containing the weights and bias
        """
        if self.weights is None or self.bias is None:
            raise ValueError(
                "Model has not been trained. Call fit() before get_coefficients()."
            )

        return {"weights": self.weights, "bias": self.bias}

    def get_training_history(self) -> Dict[str, Any]:
        """
        Get training history.

        Returns:
            Dictionary containing the loss history
        """
        return {"loss_history": self.loss_history}
