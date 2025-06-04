import numpy as np
import pytest

from algorithms.supervised.naive_bayes_classificator import NaiveBernoulliClassifier
from schemas.configs.naive_bayes_config import NaiveBayesParams

def test_fit_and_predict_simple():
    # Simple AND-like problem
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])

    model = NaiveBernoulliClassifier(NaiveBayesParams(alpha=1.0, verbose=False))
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert np.all(np.isin(preds, [0, 1]))
    # Model should predict at least the last sample correctly
    assert preds[-1] == 1

def test_predict_proba_shape_and_sum():
    X = np.array([
        [0, 1],
        [1, 0]
    ])
    y = np.array([0, 1])
    model = NaiveBernoulliClassifier()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (2, 2)
    np.testing.assert_almost_equal(probs.sum(axis=1), np.ones(2))

def test_score_keys_and_ranges():
    X = np.array([
        [0, 0],
        [1, 1]
    ])
    y = np.array([0, 1])
    model = NaiveBernoulliClassifier()
    model.fit(X, y)
    scores = model.score(X, y)
    assert "accuracy" in scores
    assert "log_loss" in scores
    assert 0 <= scores["accuracy"] <= 1
    assert scores["log_loss"] >= 0

def test_input_validation_raises():
    X = np.array([
        [0, 0],
        [1, 0]
    ])
    y_bad = np.array([0, 2])

    model = NaiveBernoulliClassifier()
    with pytest.raises(ValueError):
        model.fit(X, y_bad)

    X_bad = np.array([
        [0, 0],
        [2, 1]
    ])
    y = np.array([0, 1])
    with pytest.raises(ValueError):
        model.fit(X_bad, y)

def test_get_parameters_result():
    X = np.array([
        [1, 0],
        [0, 1]
    ])
    y = np.array([1, 0])
    model = NaiveBernoulliClassifier()
    model.fit(X, y)
    params = model.get_parameters()
    assert "class_probs" in params
    assert "feature_probs" in params
    assert "alpha" in params
    assert isinstance(params["class_probs"], np.ndarray)
    assert isinstance(params["feature_probs"], np.ndarray)