"""
This module contains implementation of the Naive Bernoulli Classifier algorithm.
"""
import logging
from typing import Any, Dict, Optional

import numpy as np
from algorithms.base.supervised import SupervisedAlgorithm
from schemas.configs.naive_bayes_config import NaiveBayesParams
from utils.losses import LossFunction, LogLoss


class NaiveBernoulliClassifier(SupervisedAlgorithm[NaiveBayesParams]):
    """
    Naive Bernoulli Classifier implementation.

    This class implements a Naive Bernoulli Classifier for binary classification tasks.
    """

    def __init__(self, params: Optional[NaiveBayesParams] = None):
        """
        Initialize the Naive Bernoulli Classifier model.

        Args:
            params: Configuration parameters for the model. Uses defaults if None.
        """
        super().__init__()
        self._params = params if params is not None else NaiveBayesParams()
        self._loss_fn: LossFunction = LogLoss()
        self.feature_probs: Optional[np.ndarray] = None
        self.class_probs: Optional[np.ndarray] = None

        # Set up logger based on verbose setting
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if self.params.verbose else logging.WARNING)

    @property
    def params(self) -> NaiveBayesParams:
        """Returns the parameters for this algorithm instance."""
        return self._params

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBernoulliClassifier":
        """
        Fit the Naive Bernoulli Classifier to the training data.

        Args:
            X: Training data features, shape (n_samples, n_features). Values should be binary (0 or 1).
            y: Target values, shape (n_samples,). Values should be binary (0 or 1).

        Returns:
            Self reference for method chaining.
        """
        n_samples, n_features = X.shape
        if n_samples == 0:
            self.logger.warning("Fitting skipped: Received empty training data.")
            return self

        # Validate that the data is binary
        if not np.array_equal(X, X.astype(bool)):
            raise ValueError(
                "Input features (X) must be binary (0 or 1). "
                "Non-binary values detected in the input data."
            )
        if not np.array_equal(y, y.astype(bool)):
            raise ValueError(
                "Target values (y) must be binary (0 or 1). "
                "Non-binary values detected in the target data."
            )

        # Calculate class probabilities (P(y=0) and P(y=1))
        classes, class_counts = np.unique(y, return_counts=True)
        self.class_probs = class_counts / n_samples

        # Use smoothing parameter from params (default to 1.0 if not present)
        alpha = getattr(self.params, "alpha", 1.0)

        # Calculate feature probabilities P(x_i=1 | y=c) for each class c, using alpha
        self.feature_probs = np.zeros((len(classes), n_features))
        for idx, c in enumerate(classes):
            X_class = X[y == c]
            # Laplace (additive) smoothing with alpha
            self.feature_probs[idx] = (np.sum(X_class, axis=0) + alpha) / (
                X_class.shape[0] + 2 * alpha
            )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for each class.

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            Probabilities for each class, shape (n_samples, n_classes).
        """
        if self.feature_probs is None or self.class_probs is None:
            raise ValueError("Model has not been trained. Call fit() before predict_proba().")

        log_class_probs = np.log(self.class_probs)
        log_feature_probs = np.log(self.feature_probs)
        log_feature_complement_probs = np.log(1 - self.feature_probs)

        log_probs = []
        for x in X:
            log_prob_c = (
                log_class_probs
                + np.sum(x * log_feature_probs, axis=1)
                + np.sum((1 - x) * log_feature_complement_probs, axis=1)
            )
            log_probs.append(log_prob_c)

        log_probs = np.vstack(log_probs)
        probs = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the input data.

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            Predicted class labels, shape (n_samples,).
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate the performance metrics for the model.

        Args:
            X: Test features, shape (n_samples, n_features).
            y: True target values, shape (n_samples,).

        Returns:
            Dictionary containing scores.
        """
        if self.feature_probs is None or self.class_probs is None:
            raise ValueError("Model has not been trained. Call fit() before score().")

        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        # For binary case (n_classes == 2) use probability of class 1
        if y_pred_proba.shape[1] == 2:
            log_loss = self._loss_fn(y, y_pred_proba[:, 1])
        else:
            log_loss = self._loss_fn(y, y_pred_proba)

        scores = {
            "log_loss": log_loss,
            "accuracy": np.mean(y == y_pred),
        }
        return scores

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the learned parameters (class and feature probabilities).

        Returns:
            Dictionary containing the learned parameters.
        """
        if self.feature_probs is None or self.class_probs is None:
            raise ValueError("Model has not been trained. Call fit() before get_parameters().")

        return {
            "class_probs": self.class_probs.copy(),
            "feature_probs": self.feature_probs.copy(),
            "alpha": getattr(self.params, "alpha", 1.0)
        }