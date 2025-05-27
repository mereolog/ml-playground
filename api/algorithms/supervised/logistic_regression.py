"""
This module contains implementation of the Logistic Regression algorithm.
"""
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from api.algorithms.base.supervised import SupervisedAlgorithm
from schemas.configs.logistic_regression import LogisticRegressionParams
from utils.losses import BinaryCrossEntropy
from utils.metrics import accuracy_score, log_loss


class LogisticRegression(SupervisedAlgorithm):
    """
    Logistic Regression implementation using gradient descent.
    """
    def __init__(self, params: Optional[LogisticRegressionParams] = None):
        super().__init__()
        self._params = params if params is not None else LogisticRegressionParams()
        self._loss_fn = BinaryCrossEntropy() 
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[float] = None
        self.loss_history: List[float] = []
        self._preprocessor: Optional[ColumnTransformer] = None
        
        self.logger = logging.getLogger(__name__)
        if self.params.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

    @property
    def params(self) -> LogisticRegressionParams:
        return self._params

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def _initialize_parameters(self, n_features: int) -> None:
        if self.params.random_state is not None:
            np.random.seed(self.params.random_state)
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        self.logger.info("Model parameters initialized.")

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise ValueError("Model parameters (weights/bias) are not initialized.")
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        n_samples = X.shape[0]
        if n_samples == 0:
            return 0.0
        if self.weights is None:
            raise ValueError("Cannot compute loss, weights are not initialized.")
        y_pred = self._predict_raw(X)
        base_loss = self._loss_fn(y, y_pred)

        reg_term = 0.0
        if self.params.regularization == "l2":
            reg_term = (self.params.lambda_ / (2 * n_samples)) * np.sum(np.square(self.weights))
        elif self.params.regularization == "l1":
            reg_term = (self.params.lambda_ / n_samples) * np.sum(np.abs(self.weights))

        return base_loss + reg_term

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
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

        if self.params.regularization == "l2":
            dw += (self.params.lambda_ / n_samples) * self.weights
        elif self.params.regularization == "l1":
            dw += (self.params.lambda_ / n_samples) * np.sign(self.weights)

        return dw, db

    def _update_parameters(self, dw: np.ndarray, db: float) -> None:
        if self.weights is None or self.bias is None:
            raise ValueError("Cannot update parameters, model not initialized.")
        self.weights -= self.params.learning_rate * dw
        self.bias -= self.params.learning_rate * db

    def _fit_preprocessor(self, X: np.ndarray) -> np.ndarray:
        if np.issubdtype(X.dtype, np.number):
            self._preprocessor = None
            return X
        self._preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(range(X.shape[1])))
            ]
        )
        return self._preprocessor.fit_transform(X)

    def _transform_preprocessor(self, X: np.ndarray) -> np.ndarray:
        if self._preprocessor is None:
            return X
        return self._preprocessor.transform(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        n_samples, n_features = X.shape
        if n_samples == 0:
            self.logger.warning("Fitting skipped: Received empty training data.")
            return self

        X = self._fit_preprocessor(X)
        self._initialize_parameters(X.shape[1])
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
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained. Call fit() before predict().")
        X = self._transform_preprocessor(X)
        y_pred_prob = self._predict_raw(X)
        return (y_pred_prob >= self.params.threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained. Call fit() before score().")
        X = self._transform_preprocessor(X)
        y_pred = self.predict(X)
        y_pred_prob = self._predict_raw(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "log_loss": log_loss(y, y_pred_prob)
        }

    def get_coefficients(self) -> Dict[str, Any]:
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained. Call fit() before get_coefficients().")
        return {"weights": self.weights.copy(), "bias": self.bias}

    def get_training_history(self) -> Dict[str, List[float]]:
        return {"training_loss": self.loss_history}

    def _get_batch_indices(self, n_samples: int, batch_size: int) -> np.ndarray:
        local_rng = np.random.RandomState(self.params.random_state)
        indices = local_rng.permutation(n_samples)
        return indices[:batch_size]
