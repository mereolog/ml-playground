from typing import List, Optional
import numpy as np


from algorithms.base.algorithm import Algorithm
from algorithms.base.unsupervised import UnsupervisedAlgorithm
from schemas.configs.algorithm_configs import UnsupervisedAlgorithmsParams 
from schemas.configs.hierarchical_clastering_config import HierarchicalClusteringParams

class HierarchicalClustering(UnsupervisedAlgorithm):
    def __init__(self, params: Optional[HierarchicalClusteringParams] = None):
        super().__init__()
        if params is None:
            params = HierarchicalClusteringParams()
        self._params = params
        self.labels_ = None

    @property
    def params(self):
        return self._params

    def _point_distance(self, a, b):
        if self._params.metric == "manhattan":
            return float(np.abs(a - b).sum())
        if self._params.metric == "cosine":
            num = float(np.dot(a, b))
            den = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
            return 1.0 - num / den
        if self._params.metric == "chebyshev":
            return float(np.abs(a - b).max())
        if self._params.metric == "minkowski":
            p = 3
            return float((np.abs(a - b) ** p).sum() ** (1.0 / p))
        return float(np.linalg.norm(a - b))

    def _cluster_distance(self, cluster_a, cluster_b, data):
        dists = []
        for i in cluster_a:
            for j in cluster_b:
                dists.append(self._point_distance(data[i], data[j]))
        if self._params.linkage == "single":
            return min(dists)
        if self._params.linkage == "complete":
            return max(dists)
        return sum(dists) / len(dists)

    def fit(self, X: np.ndarray):
        clusters = [[i] for i in range(len(X))]
        while True:
            if self._params.n_clusters is not None and len(clusters) <= self._params.n_clusters:
                break
            best_distance = float("inf")
            best_pair = None
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    d = self._cluster_distance(clusters[i], clusters[j], X)
                    if d < best_distance:
                        best_distance = d
                        best_pair = (i, j)
            if self._params.distance_threshold is not None and best_distance > self._params.distance_threshold:
                break
            if best_pair is None:
                break
            i, j = best_pair
            clusters[i].extend(clusters[j])
            clusters.pop(j)
        labels = np.empty(len(X), dtype=int)
        for cid, cluster in enumerate(clusters):
            for idx in cluster:
                labels[idx] = cid
        self.labels_ = labels.tolist()
        return self

    def fit_predict(self, X: np.ndarray):
        return self.fit(X).labels_

    def predict(self, X: np.ndarray):
        raise NotImplementedError
