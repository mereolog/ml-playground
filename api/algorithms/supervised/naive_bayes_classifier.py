"""
This module contains implementation of the Naive Bernoulli Classifier algorithm.
"""
import logging
from typing import Any, Dict, Optional

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from api.algorithms.base.supervised import SupervisedAlgorithm
from api.schemas.configs.naive_bayes_config import NaiveBayesParams
from api.utils.losses import LossFunction, LogLoss


class NaiveBernoulliClassifier(SupervisedAlgorithm[NaiveBayesParams]):
    """
    Naive Bernoulli Classifier implementation.

    This class implements a Naive Bernoulli Classifier for binary classification tasks.
    """

    def __init__(self, params: Optional[NaiveBayesParams] = None):
        super().__init__()
        self._params = params if params is not None else NaiveBayesParams()
        self._loss_fn: LossFunction = LogLoss()
        self.feature_probs: Optional[np.ndarray] = None
        self.class_probs: Optional[np.ndarray] = None
        self.logger = logging.getLogger(__name__)
        if self.params.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

    @property
    def params(self) -> NaiveBayesParams:
        return self._params

    def _check_binary(self, X: np.ndarray):
        if not np.all(np.logical_or(X == 0, X == 1)):
            raise ValueError("NaiveBernoulliClassifier przyjmuje tylko dane binarne (0 lub 1). Znaleziono inne wartości.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBernoulliClassifier":
        self._check_binary(X)
        self._check_binary(y)
        n_samples, n_features = X.shape
        if n_samples == 0:
            self.logger.warning("Fitting skipped: Received empty training data.")
            return self

        classes, class_counts = np.unique(y, return_counts=True)
        self.class_probs = np.zeros(2)
        for cls, count in zip(classes, class_counts):
            self.class_probs[int(cls)] = count / n_samples
        self.logger.info(f"Class probabilities: {self.class_probs}")

        self.feature_probs = np.zeros((2, n_features))
        for c in [0, 1]:
            X_class = X[y == c]
            if X_class.shape[0] == 0:
                self.feature_probs[c] = 0.5 * np.ones(n_features)
            else:
                self.feature_probs[c] = (np.sum(X_class, axis=0) + 1) / (X_class.shape[0] + 2)
            self.logger.info(f"Feature probabilities for class {c}: {self.feature_probs[c]}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_binary(X)
        if self.feature_probs is None or self.class_probs is None:
            raise ValueError("Model has not been trained. Call fit() before predict_proba().")

        log_class_probs = np.log(self.class_probs + 1e-12)
        log_feature_probs = np.log(self.feature_probs + 1e-12)
        log_feature_complement_probs = np.log(1 - self.feature_probs + 1e-12)
        n_samples = X.shape[0]
        n_classes = self.feature_probs.shape[0]
        log_probs = np.zeros((n_samples, n_classes))
        for c in range(n_classes):
            log_probs[:, c] = (
                log_class_probs[c]
                + np.sum(X * log_feature_probs[c], axis=1)
                + np.sum((1 - X) * log_feature_complement_probs[c], axis=1)
            )
        probs = np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_binary(X)
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        self._check_binary(X)
        self._check_binary(y)
        if self.feature_probs is None or self.class_probs is None:
            raise ValueError("Model has not been trained. Call fit() before score().")
        y_pred = self.predict(X)
        y_true = y
        log_loss_value = self._loss_fn(y_true, self.predict_proba(X))
        accuracy = np.mean(y_true == y_pred)
        scores = {
            "log_loss": log_loss_value,
            "accuracy": accuracy,
        }
        return scores

    def get_parameters(self) -> Dict[str, Any]:
        if self.feature_probs is None or self.class_probs is None:
            raise ValueError("Model has not been trained. Call fit() before get_parameters().")
        return {
            "class_probs": self.class_probs.copy(),
            "feature_probs": self.feature_probs.copy(),
        }

    def visualize_probabilities(self, X: Optional[np.ndarray] = None, show: bool = True):
        """
        Visualize class probabilities and feature probabilities using Plotly.
        If X is provided, also visualize predicted probabilities for these samples.

        Args:
            X: Optional[np.ndarray], input samples to visualize predicted probabilities.
            show: bool, whether to show the plot (default: True).
        Returns:
            Plotly Figure object.
        """
        if self.feature_probs is None or self.class_probs is None:
            raise ValueError("Model has not been trained. Call fit() before visualization.")

        rows = 2 if X is not None else 1
        fig = make_subplots(
            rows=rows, cols=1,
            subplot_titles=(
                ["Class probabilities and feature probabilities"] +
                (["Predicted probabilities for provided samples"] if X is not None else [])
            )
        )

        # 1. Class probabilities as bar
        fig.add_trace(
            go.Bar(
                x=[str(i) for i in range(len(self.class_probs))],
                y=self.class_probs,
                name='Class probabilities',
                marker=dict(color=['#636EFA', '#EF553B']),
            ),
            row=1, col=1
        )

        # 2. Feature probabilities for each class as grouped bar
        n_features = self.feature_probs.shape[1]
        for c in range(self.feature_probs.shape[0]):
            fig.add_trace(
                go.Bar(
                    x=[f"f{i}" for i in range(n_features)],
                    y=self.feature_probs[c],
                    name=f'P(x_i=1|y={c})',
                    opacity=0.6 if c == 0 else 0.9,
                ),
                row=1, col=1
            )

        # 3. If X provided, plot predicted probabilities for each sample
        if X is not None:
            probs = self.predict_proba(X)
            for i, p in enumerate(probs):
                fig.add_trace(
                    go.Bar(
                        x=[str(c) for c in range(probs.shape[1])],
                        y=p,
                        name=f"Sample {i} predicted",
                        showlegend=False,
                        marker=dict(line=dict(width=1, color='black')),
                    ),
                    row=2, col=1
                )
            fig.update_yaxes(title_text="Predicted probability", row=2, col=1)
            fig.update_xaxes(title_text="Class", row=2, col=1)

        fig.update_layout(
            barmode='group',
            title="Naive Bernoulli Classifier Probabilities Visualization",
            xaxis_title="Class / Feature",
            yaxis_title="Probability",
            legend_title="Legend",
            bargap=0.2,
            height=800 if X is not None else 500,
        )

        if show:
            fig.show()
        return fig