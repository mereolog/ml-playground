import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

from algorithms.supervised.knn import KNearestNeighbor
from schemas.configs.k_nearest_neighbour_algorithm import KNeighborsParams


@pytest.fixture
def iris_data():
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_default_params():
    params = KNeighborsParams()
    assert params.n_neighbors == 5
    assert params.weights == 'uniform'
    assert params.algorithm == 'auto'
    assert params.metric == 'minkowski'


def test_invalid_params():
    with pytest.raises(ValueError):
        KNeighborsParams(n_neighbors=0)

    with pytest.raises(ValueError):
        KNeighborsParams(p=0)

    with pytest.raises(ValueError):
        KNeighborsParams(leaf_size=0)


def test_knn_fit_predict_basic(iris_data):
    X_train, X_test, y_train, y_test = iris_data

    params = KNeighborsParams(n_neighbors=3)
    model = KNearestNeighbor(params)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert set(predictions).issubset(set(np.unique(y_train)))


def test_knn_score_matches_sklearn(iris_data):
    X_train, X_test, y_train, y_test = iris_data

    custom_knn = KNearestNeighbor(KNeighborsParams(n_neighbors=3))
    custom_knn.fit(X_train, y_train)
    score_custom = custom_knn.score(X_test, y_test)

    skl_knn = SklearnKNN(n_neighbors=3)
    skl_knn.fit(X_train, y_train)
    skl_score = skl_knn.score(X_test, y_test)

    assert abs(score_custom["accuracy"] - skl_score) < 0.1


def test_knn_predict_on_unfitted_model():
    model = KNearestNeighbor(KNeighborsParams())
    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, 2.0, 3.0, 4.0]]))


def test_knn_score_on_unfitted_model():
    model = KNearestNeighbor(KNeighborsParams())
    with pytest.raises(ValueError):
        model.score(np.array([[1.0, 2.0, 3.0, 4.0]]), np.array([0]))
