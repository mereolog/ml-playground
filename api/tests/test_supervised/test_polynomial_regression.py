import numpy as np
import pytest
from api.algorithms.supervised.polynomial_regression import PolynomialRegression
from schemas.configs.polynomial_regression import PolynomialRegressionParams


def test_polynomial_regression_on_quadratic_data():
    """
    Test polynomial regression on quadratic data.
    Model should perfectly fit a parabola when degree=2.
    """
    np.random.seed(42)
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    y = X.flatten() ** 2 + np.random.normal(0, 0.5, size=100)

    params = PolynomialRegressionParams(
        learning_rate=0.01,
        epochs=1000,
        batch_size=None,
        degree=2,
        verbose=False
    )
    model = PolynomialRegression(params=params)
    model.fit(X, y)

    predictions = model.predict(X)

    from utils.metrics import mean_squared_error, r2_score

    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    assert mse < 1.0, f"Expected low MSE (<1.0), got {mse}"
    assert r2 > 0.9, f"Expected high R² (>0.9), got {r2}"


def test_polynomial_regression_with_invalid_degree():
    """
    Test that invalid degree raises ValueError.
    """
    with pytest.raises(ValueError):
        PolynomialRegressionParams(degree=0)


def test_polynomial_regression_without_training():
    """
    Test that predict() fails before training.
    """
    model = PolynomialRegression(PolynomialRegressionParams(degree=2))
    X = np.array([[1], [2]])
    with pytest.raises(ValueError):
        model.predict(X)