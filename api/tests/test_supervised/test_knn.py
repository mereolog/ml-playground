# pytest tests/test_supervised/test_knn.py -v
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from algorithms.supervised.knn import KNearestNeighbor
from schemas.configs.k_nearest_neighbour_algorithm import KNeighborsParams


def get_default_params(**overrides):
    base = {
        "test_size": 0.2,
        "validation_size": None,
        "shuffle": True,
        "stratify": False,
        "n_neighbors": 3,
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 2,
        "metric": "minkowski",
    }
    base.update(overrides)
    return KNeighborsParams(**base)


@pytest.fixture
def iris_data():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def test_fit_and_predict_uniform(iris_data):
    X_train, X_test, y_train, y_test = iris_data
    params = get_default_params(weights="uniform")
    model = KNearestNeighbor(params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert predictions.shape == y_test.shape


def test_fit_and_predict_distance(iris_data):
    X_train, X_test, y_train, y_test = iris_data
    params = get_default_params(weights="distance")
    model = KNearestNeighbor(params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    assert predictions.shape == y_test.shape


def test_score_matches_sklearn(iris_data):
    X_train, X_test, y_train, y_test = iris_data
    params = get_default_params(n_neighbors=3)
    model = KNearestNeighbor(params)
    model.fit(X_train, y_train)
    custom_score = model.score(X_test, y_test)["accuracy"]

    ref_model = KNeighborsClassifier(n_neighbors=3)
    ref_model.fit(X_train, y_train)
    ref_score = ref_model.score(X_test, y_test)

    assert abs(custom_score - ref_score) < 0.1


def test_unfitted_model_predict_raises():
    model = KNearestNeighbor(get_default_params())
    with pytest.raises(ValueError):
        model.predict(np.array([[1.0, 2.0, 3.0, 4.0]]))


def test_unfitted_model_score_raises():
    model = KNearestNeighbor(get_default_params())
    with pytest.raises(ValueError):
        model.score(np.array([[1.0, 2.0, 3.0, 4.0]]), np.array([1]))


def test_model_with_different_metrics(iris_data):
    X_train, X_test, y_train, y_test = iris_data
    for metric in ["euclidean", "manhattan", "chebyshev"]:
        params = get_default_params(metric=metric)
        model = KNearestNeighbor(params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert len(preds) == len(y_test), f"Prediction shape mismatch for metric: {metric}"


def test_model_get_coefficients_returns_empty_dict(iris_data):
    X_train, _, y_train, _ = iris_data
    model = KNearestNeighbor(get_default_params())
    model.fit(X_train, y_train)
    coeffs = model.get_coefficients()
    assert isinstance(coeffs, dict)
    assert coeffs == {}, "Expected empty coefficients for KNN"


def test_model_get_training_history_returns_empty_dict(iris_data):
    X_train, _, y_train, _ = iris_data
    model = KNearestNeighbor(get_default_params())
    model.fit(X_train, y_train)
    history = model.get_training_history()
    assert isinstance(history, dict)
    assert history == {}, "Expected empty training history for KNN"


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        get_default_params(n_neighbors=0)


def test_invalid_p_metric_combination():
    with pytest.raises(ValueError):
        get_default_params(p=0, metric="minkowski")
