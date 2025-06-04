# tests/algorithms/supervised/test_linear_regression.py
"""Tests for the LinearRegression algorithm."""

import numpy as np
import pytest

from algorithms.supervised.linear_regression import LinearRegression


@pytest.fixture(name="model")
def fixture_model():
    """Create a LinearRegression model instance for testing."""
    return LinearRegression()


def test_loss_fn_mse(model, simple_linear_dataset):
    """Test mean squared error loss function."""
    X, y = simple_linear_dataset
    predictions = np.array([1, 2, 3])
    targets = np.array([1.5, 2.5, 3.5])
    loss = model.params._loss_fn(predictions, targets)
    expected_loss = np.mean((predictions - targets) ** 2)
    assert np.isclose(loss, expected_loss)


def test_linear_regression_no_regularization(simple_linear_dataset):
    """Test linear regression training without regularization."""
    X, y = simple_linear_dataset
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)
    assert mse < 0.1


def test_linear_regression_l1_regularization(simple_linear_dataset):
    """Test linear regression training with L1 regularization."""
    X, y = simple_linear_dataset
    model = LinearRegression()
    model.set_params(reg_type="l1", reg_strength=0.1)
    model.fit(X, y)
    _ = model.predict(X)  # Using prediction to ensure model works after training


def test_linear_regression_l2_regularization(simple_linear_dataset):
    """Test linear regression training with L2 regularization."""
    X, y = simple_linear_dataset
    model = LinearRegression()
    model.set_params(reg_type="l2", reg_strength=0.1)
    model.fit(X, y)
    _ = model.predict(X)


def test_linear_regression_elasticnet_regularization(simple_linear_dataset):
    """Test linear regression training with ElasticNet regularization."""
    X, y = simple_linear_dataset
    model = LinearRegression()
    model.set_params(
        reg_type="elasticnet",
        reg_strength=0.1,
        mixing_ratio=0.5
    )
    model.fit(X, y)
    _ = model.predict(X)
