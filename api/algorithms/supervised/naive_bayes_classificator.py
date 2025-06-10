"""
This module contains implementation of the Naive Bernoulli Classifier algorithm.
"""
import logging
from typing import Any, Dict, Optional, List

import numpy as np
import plotly.graph_objects as go

from algorithms.base.supervised import SupervisedAlgorithm
from schemas.configs.naive_bayes_config import NaiveBayesParams
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
        self._loss_fn = LogLoss()
        self.feature_probs: Optional[np.ndarray] = None
        self.class_probs: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

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

        self.classes_, class_counts = np.unique(y, return_counts=True)
        if self.classes_.tolist() != [0, 1]:
            raise ValueError("This implementation supports only binary classes labeled 0 and 1.")

        self.class_probs = class_counts / n_samples
        self.logger.info(f"Class probabilities: {self.class_probs}")

        alpha = getattr(self.params, "alpha", 1.0)

        self.feature_probs = np.zeros((2, n_features))
        for idx, c in enumerate(self.classes_):
            X_class = X[y == c]
            self.feature_probs[idx] = (np.sum(X_class, axis=0) + alpha) / (
                X_class.shape[0] + 2 * alpha
            )
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

        log_class_probs = np.log(self.class_probs)
        log_feature_probs = np.log(self.feature_probs)
        log_feature_complement_probs = np.log(1 - self.feature_probs)

        X = X.astype(np.int32)
        log_prob_c = (
            log_class_probs[None, :]
            + (X @ log_feature_probs.T)
            + ((1 - X) @ log_feature_complement_probs.T)
        )

        log_probs = log_prob_c
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
        return self.classes_[np.argmax(probs, axis=1)]

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
        if y_pred_proba.shape[1] == 2:
            idx1 = np.where(self.classes_ == 1)[0][0]
            log_loss = self._loss_fn(y, y_pred_proba[:, idx1])
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

    def plot_feature_probs(self, feature_names: Optional[List[str]] = None):
        """
        Visualize the learned feature probabilities for each class using Plotly.
        Args:
            feature_names: Optional list of feature names (length must match n_features).
        Returns:
            Plotly Figure object.
        """
        if self.feature_probs is None:
            raise ValueError("Model has not been trained. Call fit() before plotting.")

        n_classes, n_features = self.feature_probs.shape
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(n_features)]

        fig = go.Figure()
        for class_idx in range(n_classes):
            fig.add_trace(
                go.Bar(
                    x=feature_names,
                    y=self.feature_probs[class_idx],
                    name=f"Class {self.classes_[class_idx]}",
                )
            )
        fig.update_layout(
            barmode="group",
            title="Feature Probabilities per Class",
            xaxis_title="Features",
            yaxis_title="P(x_i=1 | y=class)",
            legend_title="Class",
        )
        fig.show()
        return fig