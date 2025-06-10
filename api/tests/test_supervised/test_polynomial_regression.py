#pytest tests/test_supervised/test_polynomial_regression.py
import numpy as np
import pytest

from algorithms.supervised.polynomial_regression import PolynomialRegression
from schemas.configs.polynomial_regression import PolynomialRegressionParams


def generate_data(n_samples=100):
    X = np.linspace(0, 10, n_samples).reshape(-1, 1)
    y = 3 * X.squeeze() ** 2 + 2 * X.squeeze() + 1  
    return X, y


def test_fit_runs_and_loss_decreases():
    X, y = generate_data()
    params = PolynomialRegressionParams(epochs=100, learning_rate=0.01, degree=2)
    model = PolynomialRegression(params)
    model.fit(X, y)

    history = model.get_training_history()["training_loss"]
    assert len(history) > 0
    assert history[0] > history[-1], "Loss should decrease after training."


def test_predict_shape_matches_input():
    X, y = generate_data()
    model = PolynomialRegression(PolynomialRegressionParams(degree=2))
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape


def test_predict_before_fit_raises():
    X, _ = generate_data()
    model = PolynomialRegression()
    with pytest.raises(ValueError):
        model.predict(X)


def test_get_coefficients_keys():
    X, y = generate_data()
    model = PolynomialRegression(PolynomialRegressionParams(degree=3))
    model.fit(X, y)
    coeffs = model.get_coefficients()
    assert "weights" in coeffs
    assert "bias" in coeffs
    assert "degree" in coeffs


def test_fit_with_high_degree_does_not_crash():
    X, y = generate_data()
    model = PolynomialRegression(PolynomialRegressionParams(degree=6, epochs=50))
    model.fit(X, y)
    assert len(model.get_training_history()["training_loss"]) == 50
