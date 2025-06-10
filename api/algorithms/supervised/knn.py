from algorithms.base.supervised import SupervisedAlgorithm
from schemas.configs.k_nearest_neighbour_algorithm import KNeighborsParams
import numpy as np
from typing import Optional
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter


class KNearestNeighbor(SupervisedAlgorithm):
    def __init__(self, config: KNeighborsParams):
        self.config = config
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    @property
    def params(self) -> KNeighborsParams:
        return self.config

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model not fitted yet.")

        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=self.y_train.dtype)

        distances = self._compute_distances(X)
        nearest_indices = np.argsort(distances, axis=1)[:, :self.config.n_neighbors]
        nearest_labels = self.y_train[nearest_indices]

        if self.config.weights == 'uniform':
            for i in range(n_samples):
                neighbor_labels = nearest_labels[i]
                label_counts = Counter(neighbor_labels)
                predicted_label = label_counts.most_common(1)[0][0]
                predictions[i] = predicted_label

        elif self.config.weights == 'distance':
            predictions = self._weighted_majority_vote(nearest_indices, distances, nearest_labels)
        else:
            raise ValueError(f"Unsupported weight function: {self.config.weights}")

        return predictions

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model not fitted yet.")

        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return {"accuracy": accuracy}

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        metric_map = {
            'euclidean': 'euclidean',
            'manhattan': 'manhattan',
            'chebyshev': 'chebyshev',
            'minkowski': 'minkowski'
        }

        metric = metric_map[self.config.metric]
        return pairwise_distances(X, self.X_train, metric=metric, p=self.config.p)

    def _weighted_majority_vote(self, indices: np.ndarray, distances: np.ndarray, labels: np.ndarray) -> np.ndarray:
        n_samples = distances.shape[0]
        predictions = np.zeros(n_samples, dtype=labels.dtype)

        for i in range(n_samples):
            neighbor_dists = distances[i, indices[i]]
            neighbor_labels = labels[i]
            weights = 1.0 / (neighbor_dists + 1e-8)

            label_weights = {}
            for label, weight in zip(neighbor_labels, weights):
                label_weights[label] = label_weights.get(label, 0) + weight

            predictions[i] = max(label_weights, key=label_weights.get)

        return predictions
