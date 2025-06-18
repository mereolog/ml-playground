import numpy as np
import pytest

from algorithms.supervised.naive_bayes_classificator import NaiveBernoulliClassifier
from schemas.configs.naive_bayes_config import NaiveBayesParams

def test_fit_predict_score():
    # Przygotuj dane: 4 próbki, 2 cechy, binarne
    X = np.array([
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 0]
    ])
    y = np.array([0, 1, 1, 0])

    clf = NaiveBernoulliClassifier(NaiveBayesParams(alpha=1.0, verbose=False))
    clf.fit(X, y)

    # Test predykcji
    X_test = np.array([
        [1, 0],
        [0, 1]
    ])
    preds = clf.predict(X_test)
    assert preds.shape == (2,)
    assert set(preds).issubset({0, 1})

    # Test predict_proba
    proba = clf.predict_proba(X_test)
    assert proba.shape == (2, 2)
    np.testing.assert_almost_equal(proba.sum(axis=1), np.ones(2))

    # Test score
    y_test = np.array([1, 0])
    result = clf.score(X_test, y_test)
    assert "log_loss" in result
    assert "accuracy" in result
    assert 0 <= result["accuracy"] <= 1

def test_non_binary_X_raises():
    X = np.array([
        [0, 2],
        [1, 1]
    ])
    y = np.array([0, 1])
    clf = NaiveBernoulliClassifier()
    with pytest.raises(ValueError):
        clf.fit(X, y)

def test_non_binary_y_raises():
    X = np.array([
        [0, 1],
        [1, 0]
    ])
    y = np.array([0, 2])
    clf = NaiveBernoulliClassifier()
    with pytest.raises(ValueError):
        clf.fit(X, y)

def test_only_binary_classes():
    X = np.array([
        [0, 1],
        [1, 0]
    ])
    y = np.array([0, 2])
    clf = NaiveBernoulliClassifier()
    with pytest.raises(ValueError):
        clf.fit(X, y)

def test_predict_without_fit_raises():
    clf = NaiveBernoulliClassifier()
    with pytest.raises(ValueError):
        clf.predict(np.array([[0, 1]]))

def test_predict_proba_without_fit_raises():
    clf = NaiveBernoulliClassifier()
    with pytest.raises(ValueError):
        clf.predict_proba(np.array([[0, 1]]))

def test_get_parameters_without_fit_raises():
    clf = NaiveBernoulliClassifier()
    with pytest.raises(ValueError):
        clf.get_parameters()