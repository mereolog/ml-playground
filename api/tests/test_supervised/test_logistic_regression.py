#pytest tests/test_supervised/test_logistic_regression.py -v
import numpy as np
import pytest
from algorithms.supervised.logistic_regression import LogisticRegression
from schemas.configs.logistic_regression import LogisticRegressionParams

X_simple = np.array([[0], [1], [2], [3]])
y_simple = np.array([0, 0, 1, 1])

@pytest.fixture
def simple_params():
    return LogisticRegressionParams(
        learning_rate=0.1,
        epochs=200,
        batch_size=None,
        regularization=None,
        lambda_=0.0,
        threshold=0.5
    )

def test_model_initialization(simple_params):
    model = LogisticRegression(simple_params)
    assert model.params.epochs == 200
    assert model.params.threshold == 0.5

def test_fit_and_predict(simple_params):
    model = LogisticRegression(simple_params)
    model.fit(X_simple, y_simple)
    y_pred = model.predict(X_simple)
    assert y_pred.shape == y_simple.shape
    assert np.all(np.isin(y_pred, [0, 1]))

def test_score_function(simple_params):
    model = LogisticRegression(simple_params)
    model.fit(X_simple, y_simple)
    scores = model.score(X_simple, y_simple)
    assert "accuracy" in scores
    assert "log_loss" in scores
    assert 0.0 <= scores["accuracy"] <= 1.0

def test_invalid_shape(simple_params):
    model = LogisticRegression(simple_params)
    X_bad = np.array([[1, 2], [3, 4]])  
    y_bad = np.array([0])             
    with pytest.raises(ValueError):
        model.fit(X_bad, y_bad)

def test_invalid_labels(simple_params):
    model = LogisticRegression(simple_params)
    y_invalid = np.array([0, 1, 2, 1])  # label "2" is invalid
    with pytest.raises(ValueError, match="binary labels"):
        model.fit(X_simple, y_invalid)