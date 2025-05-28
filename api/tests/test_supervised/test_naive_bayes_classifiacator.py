import numpy as np
import pytest
from schemas.configs.naive_bayes_configs import NaiveBayesParams
from algorithms.supervised.naive_bayes_classificator import NaiveBernoulliClassifier

def test_fit_predict_score_basic():
    # Simple dataset: AND logic gate
    # X: [A, B], y: A AND B
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])

    params = NaiveBayesParams(alpha=1.0, verbose=False)
    clf = NaiveBernoulliClassifier(params)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    preds = clf.predict(X)
    scores = clf.score(X, y)
    params_dict = clf.get_parameters()

    # Probabilities should be between 0 and 1
    assert np.all(probs >= 0) and np.all(probs <= 1)
    # Predictions should match y (perfect accuracy for simple AND logic)
    assert np.array_equal(preds, y)
    # Accuracy should be 1.0
    assert pytest.approx(scores["accuracy"], 0.01) == 1.0
    # Should expose learned parameters
    assert "class_probs" in params_dict and "feature_probs" in params_dict

def test_fit_with_nonbinary_raises():
    X = np.array([
        [0, 1],
        [0, 2],  # Non-binary value!
        [1, 0]
    ])
    y = np.array([0, 0, 1])
    clf = NaiveBernoulliClassifier()
    with pytest.raises(ValueError):
        clf.fit(X, y)

def test_predict_without_fit_raises():
    X = np.array([[0, 1], [1, 0]])
    clf = NaiveBernoulliClassifier()
    with pytest.raises(ValueError):
        clf.predict_proba(X)

def test_plot_feature_probs_runs(tmp_path):
    # Test that plot_feature_probs runs without errors
    X = np.array([[0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1])
    clf = NaiveBernoulliClassifier()
    clf.fit(X, y)
    # Should return a Plotly Figure
    fig = clf.plot_feature_probs()
    assert fig is not None

def test_plot_predictions_runs(tmp_path):
    # Test that plot_predictions runs without errors
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])
    clf = NaiveBernoulliClassifier()
    clf.fit(X, y)
    fig = clf.plot_predictions(X, y, feature_x=0, feature_y=1)
    assert fig is not None