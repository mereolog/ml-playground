"""Tests for the LinearRegression algorithm implementation."""

import numpy as np
import pytest
from algorithms.supervised.linear_regression import LinearRegression
from numpy.testing import assert_allclose
from schemas.configs.algorithms_configs import LinearRegressionParams


def test_something():
    assert True


class TestLinearRegression:
    """Test suite for LinearRegression algorithm."""

    def test_initialization(self):
        """Test that the model initializes correctly with default and custom parameters."""
        # Default parameters
        model = LinearRegression()
        assert model.params.learning_rate == 0.01
        assert model.params.epochs == 100
        assert model.params.regularization is None

        # Custom parameters
        custom_params = LinearRegressionParams(
            learning_rate=0.05,
            epochs=200,
            regularization=0.01,
            batch_size=32,
            random_state=42,
        )
        model = LinearRegression(params=custom_params)
        assert model.params.learning_rate == 0.05
        assert model.params.epochs == 200
        assert model.params.regularization == 0.01
        assert model.params.batch_size == 32
        assert model.params.random_state == 42

    def test_get_set_params(self):
        """Test parameter getter and setter methods."""
        model = LinearRegression()
        params = model.get_params()
        assert "learning_rate" in params
        assert "epochs" in params

        # Test setting individual parameters
        model.set_params(learning_rate=0.1, epochs=50)
        assert model.params.learning_rate == 0.1
        assert model.params.epochs == 50

        # Test invalid parameter
        with pytest.raises(ValueError):
            model.set_params(invalid_param=10)

    def test_fit_predict(self, simple_linear_dataset):
        """Test model fitting and prediction."""
        X, y, true_weights, true_bias = simple_linear_dataset

        # Create and fit model
        model = LinearRegression(
            params=LinearRegressionParams(
                learning_rate=0.1, epochs=200, random_state=42
            )
        )
        model.fit(X, y)

        # Check that model parameters are learned
        assert model.weights is not None
        assert model.bias is not None

        # Check predictions
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape

        # Check that model has learned something (loss decreased)
        assert model.loss_history[0] > model.loss_history[-1]

        # Check that weights are reasonably close to true weights
        # (within some tolerance, since this is stochastic)
        assert_allclose(model.weights, true_weights, rtol=0.2, atol=0.2)
        assert abs(model.bias - true_bias) < 0.3

    def test_score(self, simple_linear_dataset):
        """Test the scoring method."""
        X, y, _, _ = simple_linear_dataset

        # Create and fit model
        model = LinearRegression(
            params=LinearRegressionParams(
                learning_rate=0.1, epochs=200, random_state=42
            )
        )
        model.fit(X, y)

        # Calculate R^2 score
        r2_score = model.score(X, y)
        assert (
            0 <= r2_score <= 1
        )  # R^2 should be between 0 and 1 for a reasonable model
        assert r2_score > 0.8  # Should be a good fit for this simple dataset

    def test_batch_training(self, simple_linear_dataset):
        """Test mini-batch training."""
        X, y, _, _ = simple_linear_dataset

        # Create and fit model with mini-batch
        model = LinearRegression(
            params=LinearRegressionParams(
                learning_rate=0.1, epochs=200, batch_size=32, random_state=42
            )
        )
        model.fit(X, y)

        # Check that model parameters are learned
        assert model.weights is not None
        assert model.bias is not None

        # Check that model has learned something (loss decreased)
        assert model.loss_history[0] > model.loss_history[-1]

    def test_regularization(self, simple_linear_dataset):
        """Test that regularization affects weights."""
        X, y, _, _ = simple_linear_dataset

        # Create and fit model without regularization
        model_no_reg = LinearRegression(
            params=LinearRegressionParams(
                learning_rate=0.1, epochs=200, random_state=42
            )
        )
        model_no_reg.fit(X, y)

        # Create and fit model with regularization
        model_with_reg = LinearRegression(
            params=LinearRegressionParams(
                learning_rate=0.1, epochs=200, regularization=0.1, random_state=42
            )
        )
        model_with_reg.fit(X, y)

        # Regularized weights should have smaller magnitude
        assert np.sum(np.abs(model_with_reg.weights)) < np.sum(
            np.abs(model_no_reg.weights)
        )

    def test_get_coefficients(self, simple_linear_dataset):
        """Test retrieving model coefficients."""
        X, y, _, _ = simple_linear_dataset

        model = LinearRegression(
            params=LinearRegressionParams(
                learning_rate=0.1, epochs=100, random_state=42
            )
        )
        model.fit(X, y)

        coeffs = model.get_coefficients()
        assert "weights" in coeffs
        assert "bias" in coeffs
        assert coeffs["weights"].shape == (X.shape[1],)

    def test_untrained_errors(self):
        """Test that appropriate errors are raised when model is not trained."""
        X = np.random.rand(10, 2)
        model = LinearRegression()

        with pytest.raises(ValueError):
            model.predict(X)

        with pytest.raises(ValueError):
            model.get_coefficients()
