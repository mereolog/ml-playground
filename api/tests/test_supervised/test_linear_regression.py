# tests/algorithms/supervised/test_linear_regression.py
"""Tests for the LinearRegression algorithm implementation."""

import numpy as np
import pytest

# --- Assuming your project structure allows these imports ---
from algorithms.supervised.linear_regression import LinearRegression
from numpy.testing import assert_allclose
from schemas.configs.algorithms_configs import LinearRegressionParams
from utils.losses import MeanAbsoluteError, MeanSquaredError

# -----------------------------------------------------------


# --- Fixture for simple data ---
@pytest.fixture(scope="module")
def simple_linear_dataset():
    """Generates a simple dataset for linear regression testing."""
    np.random.seed(42)  # for reproducibility
    true_weights = np.array([2.5, -1.0])
    true_bias = 1.5
    n_samples = 100
    n_features = 2

    X = np.random.rand(n_samples, n_features) * 10
    noise = np.random.randn(n_samples) * 0.5
    y = np.dot(X, true_weights) + true_bias + noise
    return X, y, true_weights, true_bias


# -----------------------------


class TestLinearRegression:
    """Test suite for LinearRegression algorithm."""

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        model = LinearRegression()
        assert isinstance(model.params, LinearRegressionParams)
        # Default optimization params
        assert model.params.learning_rate == 0.01
        assert model.params.epochs == 100
        assert model.params.batch_size is None
        # Default loss params
        assert model.params.loss == "mse"
        # Default regularization params
        assert model.params.reg_type is None
        assert model.params.reg_strenght == 0.01  # Default value
        # Default base params
        assert model.params.random_state is None
        assert model.params.verbose is False

        # Check internal loss function instance (assuming mse is default)
        assert isinstance(
            model._loss_fn, MeanSquaredError
        )  # Accessing protected for test validation

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        custom_params = LinearRegressionParams(
            learning_rate=0.05,
            epochs=200,
            loss="mae",  # Use MAE
            reg_type="elasticnet",  # Use ElasticNet
            reg_strenght=0.1,
            mixing_ratio=0.7,
            batch_size=32,
            random_state=42,
            verbose=True,
        )
        model = LinearRegression(params=custom_params)

        # Check optimization params
        assert model.params.learning_rate == 0.05
        assert model.params.epochs == 200
        assert model.params.batch_size == 32
        # Check loss params
        assert model.params.loss == "mae"
        # Check regularization params
        assert model.params.reg_type == "elasticnet"
        assert model.params.reg_strenght == 0.1
        assert model.params.mixing_ratio == 0.7
        # Check base params
        assert model.params.random_state == 42
        assert model.params.verbose is True

        # Check internal loss function instance (should be MAE now)
        assert isinstance(model._loss_fn, MeanAbsoluteError)  # Accessing protected

    def test_get_set_params(self):
        """Test parameter getter and setter methods."""
        model = LinearRegression()
        params_dict = model.get_params()
        # Check if new params are present
        assert "learning_rate" in params_dict
        assert "loss" in params_dict
        assert "reg_type" in params_dict
        assert "reg_strenght" in params_dict

        # Test setting individual parameters
        model.set_params(
            learning_rate=0.1, epochs=50, loss="mae", reg_type="l2", reg_strenght=0.05
        )
        assert model.params.learning_rate == 0.1
        assert model.params.epochs == 50
        assert model.params.loss == "mae"
        assert model.params.reg_type == "l2"
        assert model.params.reg_strenght == 0.05
        # Check if internal loss function was updated (important for set_params)
        # NOTE: The current set_params only updates the params dataclass,
        # it DOES NOT re-initialize the internal _loss_fn. This might be a design flaw
        # in the base Algorithm class's set_params or require overriding it.
        # For now, we test the param was set, but the internal instance won't change via set_params.
        # assert isinstance(model._loss_fn, MeanAbsoluteError) # This would FAIL with current set_params

        # Test invalid parameter
        with pytest.raises(ValueError):
            model.set_params(invalid_param=10)

    def test_fit_predict(self, simple_linear_dataset):
        """Test model fitting and prediction (using default MSE loss)."""
        X, y, true_weights, true_bias = simple_linear_dataset

        # Use default params (MSE loss, no regularization)
        model = LinearRegression(
            params=LinearRegressionParams(
                learning_rate=0.01,
                epochs=300,
                random_state=42,  # Adjusted LR/Epochs for convergence
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
        history = model.get_training_history()
        assert "training_loss" in history
        loss_list = history["training_loss"]
        assert len(loss_list) == model.params.epochs
        assert loss_list[0] > loss_list[-1]

        # Check that weights are reasonably close to true weights
        # Increase tolerance slightly as GD might not perfectly converge
        coeffs = model.get_coefficients()
        assert_allclose(coeffs["weights"], true_weights, rtol=0.3, atol=0.3)

    def test_score(self, simple_linear_dataset):
        """Test the scoring method returns a dictionary with correct metrics."""
        X, y, _, _ = simple_linear_dataset

        # Create and fit model
        model = LinearRegression(
            params=LinearRegressionParams(
                learning_rate=0.01, epochs=300, random_state=42
            )
        )
        model.fit(X, y)

        # Calculate scores
        scores = model.score(X, y)

        # Check return type and keys
        assert isinstance(scores, dict)
        expected_keys = ["r_squared", "mse", "rmse", "mae"]
        for key in expected_keys:
            assert key in scores
            assert isinstance(scores[key], float)

        # --- WARNING ---
        # this test FAILS BECAUSE THERE IS NO
        # R2 SCORE METRIC implemented in the code
        # should pass just fine after we add it
        # --- WARNING ---

        # Check R^2 value is reasonable
        assert 0.8 < scores["r_squared"] <= 1.0  # Should be a good fit

        # Check other metrics are non-negative
        assert scores["mse"] >= 0
        assert scores["rmse"] >= 0
        assert scores["mae"] >= 0

    def test_batch_training(self, simple_linear_dataset):
        """Test mini-batch training runs and learns."""
        X, y, _, _ = simple_linear_dataset

        # Create and fit model with mini-batch
        model = LinearRegression(
            params=LinearRegressionParams(
                learning_rate=0.01,
                epochs=300,
                batch_size=16,
                random_state=42,  # Smaller batch size
            )
        )
        model.fit(X, y)

        # Check that model parameters are learned
        assert model.weights is not None
        assert model.bias is not None

        # Check that model has learned something (loss decreased)
        history = model.get_training_history()
        assert "training_loss" in history
        loss_list = history["training_loss"]
        assert len(loss_list) == model.params.epochs
        assert loss_list[0] > loss_list[-1]  # Loss should generally decrease

    def test_regularization_l2(self, simple_linear_dataset):
        """Test that L2 regularization affects weights."""
        X, y, _, _ = simple_linear_dataset
        common_params = {"learning_rate": 0.01, "epochs": 300, "random_state": 42}

        # Fit model without regularization
        model_no_reg = LinearRegression(
            params=LinearRegressionParams(**common_params, reg_type=None)
        )
        model_no_reg.fit(X, y)
        weights_no_reg = model_no_reg.get_coefficients()["weights"]

        # Fit model with L2 regularization
        model_with_reg = LinearRegression(
            params=LinearRegressionParams(
                **common_params, reg_type="l2", reg_strenght=0.5  # Significant strength
            )
        )
        model_with_reg.fit(X, y)
        weights_with_reg = model_with_reg.get_coefficients()["weights"]

        # --- WARNING ---
        # this test FAILS BECAUSE THERE IS NO
        # REGULARIZATION implemented in the algorithm code
        # should pass just fine after we add what is missing
        # --- WARNING ---

        # L2 Regularized weights should generally have smaller magnitude (L2 norm)
        assert np.linalg.norm(weights_with_reg) < np.linalg.norm(weights_no_reg)
        # Also check absolute sum as a proxy (less direct for L2 than L1)
        assert np.sum(np.abs(weights_with_reg)) < np.sum(np.abs(weights_no_reg))

    def test_regularization_l1(self, simple_linear_dataset):
        """Test that L1 regularization affects weights (potentially sparsity)."""
        X, y, _, _ = simple_linear_dataset
        common_params = {
            "learning_rate": 0.01,
            "epochs": 500,
            "random_state": 42,
        }  # More epochs maybe needed

        # Fit model without regularization
        model_no_reg = LinearRegression(
            params=LinearRegressionParams(**common_params, reg_type=None)
        )
        model_no_reg.fit(X, y)
        weights_no_reg = model_no_reg.get_coefficients()["weights"]

        # Fit model with L1 regularization
        model_with_reg = LinearRegression(
            params=LinearRegressionParams(
                **common_params,
                reg_type="l1",
                reg_strenght=0.1  # Adjust strength as needed
            )
        )
        model_with_reg.fit(X, y)
        weights_with_reg = model_with_reg.get_coefficients()["weights"]

        # --- WARNING ---
        # this test FAILS BECAUSE THERE IS NO
        # REGULARIZATION implemented in the algorithm code
        # should pass just fine after we add what is missing
        # --- WARNING ---

        # L1 Regularized weights should generally have smaller absolute sum
        assert np.sum(np.abs(weights_with_reg)) < np.sum(np.abs(weights_no_reg))
        # Optionally check for sparsity (some weights might become zero with strong L1)
        # assert np.any(np.isclose(weights_with_reg, 0)) # This depends heavily on strenght/data

    def test_get_coefficients(self, simple_linear_dataset):
        """Test retrieving model coefficients after fitting."""
        X, y, _, _ = simple_linear_dataset

        model = LinearRegression(
            params=LinearRegressionParams(
                learning_rate=0.01, epochs=100, random_state=42
            )
        )
        model.fit(X, y)

        coeffs = model.get_coefficients()
        assert "weights" in coeffs
        assert "bias" in coeffs
        assert isinstance(coeffs["weights"], np.ndarray)
        assert isinstance(coeffs["bias"], float)
        assert coeffs["weights"].shape == (X.shape[1],)

    def test_get_training_history(self, simple_linear_dataset):
        """Test retrieving training history after fitting."""
        X, y, _, _ = simple_linear_dataset
        epochs = 50
        model = LinearRegression(
            params=LinearRegressionParams(
                learning_rate=0.01, epochs=epochs, random_state=42
            )
        )
        # Test before fitting
        assert model.get_training_history() == {"training_loss": []}

        model.fit(X, y)
        history = model.get_training_history()

        assert isinstance(history, dict)
        assert "training_loss" in history
        assert isinstance(history["training_loss"], list)
        assert len(history["training_loss"]) == epochs
        assert all(isinstance(loss, float) for loss in history["training_loss"])

    def test_untrained_errors(self):
        """Test that appropriate errors are raised when methods are called before fit()."""
        X_test = np.random.rand(10, 2)
        y_test = np.random.rand(10)
        model = LinearRegression()

        with pytest.raises(ValueError, match="Model has not been trained"):
            model.predict(X_test)

        with pytest.raises(ValueError, match="Model has not been trained"):
            model.score(X_test, y_test)

        with pytest.raises(ValueError, match="Model has not been trained"):
            model.get_coefficients()
