from schemas.configs.algorithms_configs import (BaseAlgorithmParams,
                                                GradientBasedParams,
                                                SupervisedAlgorithmsParams)
from schemas.configs.linear_regression import LinearRegressionParams


class TestLinearRegressionParams:
    """Test suite for LinearRegressionParams."""

    def test_inheritance(self):
        """Test that LinearRegressionParams inherits from SupervisedAlgorithmsParams."""
        params = LinearRegressionParams()
        assert isinstance(params, SupervisedAlgorithmsParams)
        assert isinstance(params, BaseAlgorithmParams)
        assert isinstance(params, GradientBasedParams)

    def test_initialization(self):
        """Test initialization with default and custom values."""
        # Default initialization
        params = LinearRegressionParams()
        assert params.learning_rate == 0.01
        assert params.epochs == 100
        assert params.reg_strength == 0.01
        assert params.batch_size is None
        assert params.random_state is None
        assert params.verbose is False

        # Custom initialization
        params = LinearRegressionParams(
            learning_rate=0.05,
            epochs=200,
            reg_strength=0.01,
            batch_size=32,
            random_state=42,
            verbose=True,
        )
        assert params.learning_rate == 0.05
        assert params.epochs == 200
        assert params.reg_strength == 0.01
        assert params.batch_size == 32
        assert params.random_state == 42
        assert params.verbose is True
