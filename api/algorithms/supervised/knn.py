from collections import Counter
from typing import Optional, Dict, Any, List

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from algorithms.base.supervised import SupervisedAlgorithm
from schemas.configs.k_nearest_neighbour_algorithm import KNeighborsParams


class KNearestNeighbor(SupervisedAlgorithm[KNeighborsParams]):
    def __init__(self, config: KNeighborsParams):
        super().__init__()
        self._params = config
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    @property
    def params(self) -> KNeighborsParams:
        return self._params

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model is not fitted yet.")

        distances = self._compute_distances(X)
        nearest_indices = np.argsort(distances, axis=1)[:, :self.params.n_neighbors]
        nearest_labels = self.y_train[nearest_indices]

        if self.params.weights == "uniform":
            return self._uniform_vote(nearest_labels)
        elif self.params.weights == "distance":
            return self._distance_vote(nearest_indices, distances, nearest_labels)
        else:
            raise ValueError(f"Unsupported weights type: {self.params.weights}")

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        y_pred = self.predict(X)
        accuracy = float(np.mean(y_pred == y))
        return {"accuracy": accuracy}

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        metric = self.params.metric
        if metric == "minkowski":
            return pairwise_distances(X, self.X_train, metric=metric, p=self.params.p)
        else:
            return pairwise_distances(X, self.X_train, metric=metric)

    def _uniform_vote(self, nearest_labels: np.ndarray) -> np.ndarray:
        predictions = []
        for labels in nearest_labels:
            most_common = Counter(labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)

    def _distance_vote(self, indices: np.ndarray, distances: np.ndarray, nearest_labels: np.ndarray) -> np.ndarray:
        predictions = []
        for i in range(len(indices)):
            neighbor_dists = distances[i, indices[i]]
            neighbor_labels = nearest_labels[i]
            weights = 1.0 / (neighbor_dists + 1e-8)
            vote = {}

            for label, weight in zip(neighbor_labels, weights):
                vote[label] = vote.get(label, 0) + weight

            predictions.append(max(vote, key=vote.get))
        return np.array(predictions)

    def get_coefficients(self) -> Dict[str, Any]:
        return {}

    def get_training_history(self) -> Dict[str, List[float]]:
        return {}
