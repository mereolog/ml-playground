"""
This module contains implementation of the Naive Bernoulli Classifier algorithm.
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from api.algorithms.base.supervised import SupervisedAlgorithm
from schemas.configs.naive_bayes_configs import NaiveBayesParams
from utils.losses import LossFunction, LogLoss
from utils.losses import LogLoss


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

        # LogLoss function for calculating probabilities-based loss
        self._loss_fn: LossFunction = LogLoss()

        # Learned probabilities for each feature
        self.feature_probs: Optional[np.ndarray] = None
        self.class_probs: Optional[np.ndarray] = None

        # Set up logger based on verbose setting
        self.logger = logging.getLogger(__name__)
        if self.params.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

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

        # Calculate class probabilities (P(y=0) and P(y=1))
        classes, class_counts = np.unique(y, return_counts=True)
        self.class_probs = class_counts / n_samples
        self.logger.info(f"Class probabilities: {self.class_probs}")

        # Calculate feature probabilities P(x_i=1 | y=c) for each class c
        self.feature_probs = np.zeros((len(classes), n_features))
        for idx, c in enumerate(classes):
            X_class = X[y == c]
            self.feature_probs[idx] = (np.sum(X_class, axis=0) + 1) / (
                X_class.shape[0] + 2
            )  # Laplace smoothing
            self.logger.info(f"Feature probabilities for class {c}: {self.feature_probs[idx]}")

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

        # Logarithmic probabilities for numerical stability
        log_class_probs = np.log(self.class_probs)
        log_feature_probs = np.log(self.feature_probs)
        log_feature_complement_probs = np.log(1 - self.feature_probs)

        # Calculate log-probabilities for each class
        log_probs = []
        for x in X:
            log_prob_c = (
                log_class_probs
                + np.sum(x * log_feature_probs, axis=1)
                + np.sum((1 - x) * log_feature_complement_probs, axis=1)
            )
            log_probs.append(log_prob_c)

        return np.exp(log_probs)  # Convert back to probabilities

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

        scores = {
            "log_loss": LogLoss(y, self.predict_proba(X)),
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
        }