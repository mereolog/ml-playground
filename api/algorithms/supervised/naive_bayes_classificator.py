"""
This module contains implementation of the Naive Bernoulli Classifier algorithm.
"""
import logging
from typing import Any, Dict, Optional

import numpy as np
import plotly.express as px
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

        # Use log loss function for calculating probabilities-based loss
        self._loss_fn: LogLoss()

        # Learned probabilities for each feature
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
        self.logger.info(f"Class probabilities: {self.class_probs}")

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

        # Convert back to probabilities with normalization (softmax style for stability)
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

        scores = {
            "log_loss": self._loss_fn.compute(y, self.predict_proba(X)),
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

    def plot_feature_probs(self, feature_names: Optional[list] = None):
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
                    name=f"Class {class_idx}",
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

    def plot_predictions(self, X: np.ndarray, y: np.ndarray, feature_x: int = 0, feature_y: int = 1):
        """
        Visualize the classifier decision and true labels in 2D (for two selected features).

        Args:
            X: Feature matrix.
            y: True classes.
            feature_x: Feature index for X axis.
            feature_y: Feature index for Y axis.
        Returns:
            Plotly Figure object.
        """
        if X.shape[1] <= max(feature_x, feature_y):
            raise ValueError("Selected feature indices out of bounds.")
        preds = self.predict(X)
        fig = px.scatter(
            x=X[:, feature_x],
            y=X[:, feature_y],
            color=[str(label) for label in y],
            symbol=[str(pred) for pred in preds],
            labels={"color": "True class", "symbol": "Predicted"},
            title="True classes and predicted labels (symbols) in feature space",
        )
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
        fig.show()
        return fig