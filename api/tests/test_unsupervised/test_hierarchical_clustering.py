# tests/algorithms/unsupervised/test_hierarchical_clustering.py
import numpy as np
import pytest

from algorithms.unsupervised.hierarchical_clustering import HierarchicalClustering
from schemas.configs.hierarchical_clastering_config import HierarchicalClusteringParams


def test_two_clusters_with_euclidean_metric():
    X = np.array([[0], [1], [10], [11]])
    params = HierarchicalClusteringParams(
        n_clusters=2,
        linkage="single",
        metric="euclidean"
    )
    model = HierarchicalClustering(params)
    labels = model.fit_predict(X)

    assert len(set(labels)) == 2
    # Klasteryzacja powinna rozdzielić dane na [0,1] i [10,11]
    group1 = labels[0:2]
    group2 = labels[2:4]
    assert len(set(group1)) == 1
    assert len(set(group2)) == 1
    assert group1[0] != group2[0]


def test_distance_threshold_stops_merging():
    X = np.array([[0], [1], [10], [11]])
    params = HierarchicalClusteringParams(
        distance_threshold=1.5,
        linkage="single",
        metric="euclidean"
    )
    model = HierarchicalClustering(params)
    labels = model.fit_predict(X)

    # Każdy punkt powinien być osobnym klastrem
    assert len(set(labels)) == 4


@pytest.mark.parametrize("metric", ["manhattan", "cosine", "chebyshev", 
"minkowski", "euclidean"])
def test_all_supported_metrics(metric):
    X = np.array([[1, 2], [1, 3], [10, 10], [11, 11]])
    params = HierarchicalClusteringParams(n_clusters=2, linkage="complete", metric=metric)
    model = HierarchicalClustering(params)
    labels = model.fit_predict(X)

    assert len(set(labels)) == 2


def test_predict_raises_not_implemented():
    model = HierarchicalClustering()
    with pytest.raises(NotImplementedError):
        model.predict(np.array([[0.0]]))

