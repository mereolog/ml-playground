"""Tests for linear regression configuration classes.

This module contains test cases for validating the behavior and functionality
of the linear regression configuration parameters.
"""

from schemas.configs.algorithm_configs import GradientBasedParams
from schemas.configs.linear_regression import LinearRegressionParams


def test_linear_regression_params():
    """Test LinearRegressionParams initialization and inheritance."""
    params = LinearRegressionParams()
    
    # Test inheritance
    assert isinstance(params, GradientBasedParams)
    
    # Test default values
    assert params.learning_rate == 0.01
    assert params.epochs == 100
    assert params.batch_size is None
    assert params.reg_type is None
    assert params.reg_strength == 0.01
    assert params.mixing_ratio == 0.5
