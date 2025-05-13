"""
This module contains implementation of the Logistic Regression algorithm.
"""
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from api.algorithms.base.supervised import SupervisedAlgorithm
from schemas.configs.logistic_regression import LogisticRegressionParams
from utils.losses import BinaryCrossEntropy
from utils.metrics import accuracy_score, log_loss

class LogisticRegression(SupervisedAlgorithm):
    """
    Logistic Regression implementation using gradient descent.
    This class implements logistic regression for binary classification
    using gradient descent optimization to minimize the binary cross-entropy loss.
    """
    def __init__(self, params: Optional[LogisticRegressionParams] = None):
        """
        Initialize the logistic regression model.
        Args:
            params: Configuration parameters for the model. Uses defaults if None.
        """
        super().__init__()
        self._params = params if params is not None else LogisticRegressionParams()
        self._loss_fn = BinaryCrossEntropy() 
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.loss_history: List[float] = []
        
        self.logger = logging.getLogger(__name__)
        if self.params.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

    @property
    def params(self) -> LogisticRegressionParams:
        """Returns the parameters for this algorithm instance."""
        return self._params

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def _initialize_parameters(self, n_features: int) -> None:
        """
        Initialize model weights and bias.
        Args:
            n_features: Number of input features.
        """
        if self.params.random_state is not None:
            np.random.seed(self.params.random_state)


        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.logger.info("Model parameters initialized.")

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Make raw predictions using the linear model z = Xw + b,
        then apply sigmoid to get probabilities.
        Args:
            X: Input features, shape (n_samples, n_features).
        Returns:
            Predicted probabilities, shape (n_samples,).
        Raises:
            ValueError: If weights or bias are not initialized.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model parameters (weights/bias) are not initialized.")
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the binary cross-entropy loss.
        Args:
            X: Input features, shape (n_samples, n_features).
            y: True labels, shape (n_samples,).
        Returns:
            Scalar loss value.
        """
        n_samples = X.shape[0]
        if n_samples == 0:
            return 0.0
        if self.weights is None:
            raise ValueError("Cannot compute loss, weights are not initialized.")
        y_pred = self._predict_raw(X)
        base_loss = self._loss_fn(y, y_pred)

        reg_strength = self.params.reg_strength
        reg_type = self.params.reg_type
        # reg_penalty = 

        return base_loss  # + reg_penalty

    def _compute_gradients(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weights and bias using binary cross-entropy.
        Args:
            X: Input features, shape (n_samples, n_features).
            y: True labels, shape (n_samples,).
        Returns:
            Tuple of (weight gradients, bias gradient).
        """
        n_samples = X.shape[0]
        if n_samples == 0:
            if self.weights is not None:
                return np.zeros_like(self.weights), 0.0
            else:
                raise ValueError("Cannot compute gradients, weights not initialized.")
        if self.weights is None:
            raise ValueError("Cannot compute gradients, weights are not initialized.")

        y_pred = self._predict_raw(X)
        
        error = y_pred - y
        dw = np.dot(X.T, error) / n_samples
        db = np.sum(error) / n_samples

        
        reg_strength = self.params.reg_strength
        reg_type = self.params.reg_type
        

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Fit the logistic regression model to the training data using gradient descent.
        Args:
            X: Training data features, shape (n_samples, n_features).
            y: Target labels, shape (n_samples,).
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
            
                dw, db = self._compute_gradients(X, y)
                self._update_parameters(dw, db)
            else:
                
                for i in range(0, n_samples, actual_batch_size):
                    batch_indices = self._get_batch_indices(n_samples, actual_batch_size)
                    X_batch = X[batch_indices]
                    y_batch = y[batch_indices]
                    if X_batch.shape[0] == 0:
                        continue
                    dw, db = self._compute_gradients(X_batch, y_batch)
                    self._update_parameters(dw, db)

            
            current_loss = self._compute_loss(X, y)
            self.loss_history.append(current_loss)

            
            if self.params.verbose and (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.params.epochs}, Loss: {current_loss:.6f}"
                )

        self.logger.info(f"Training finished after {self.params.epochs} epochs.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using the trained logistic regression model.
        Args:
            X: Input features, shape (n_samples, n_features).
        Returns:
            Predicted class labels (0 or 1), shape (n_samples,).
        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained. Call fit() before predict().")
        y_pred_prob = self._predict_raw(X)
        return (y_pred_prob >= self.params.threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics for the model.
        Args:
            X: Test features, shape (n_samples, n_features).
            y: True target labels, shape (n_samples,).
        Returns:
            Dictionary containing metrics like accuracy and log loss.
        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained. Call fit() before score().")
        y_pred = self.predict(X)
        y_pred_prob = self._predict_raw(X)
        scores = {
            "accuracy": accuracy_score(y, y_pred),
            "log_loss": log_loss(y, y_pred_prob)
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

    def _get_batch_indices(self, n_samples: int, batch_size: int) -> np.ndarray:
        """
        Get random batch indices. Handles random state.
        Args:
            n_samples: Total number of samples.
            batch_size: Size of the batch.
        Returns:
            Array of random indices.
        """
        local_rng = np.random.RandomState(self.params.random_state)
        indices = local_rng.permutation(n_samples)
        return indices[:batch_size]