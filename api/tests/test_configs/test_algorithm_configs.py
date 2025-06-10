"""Tests for algorithm configuration dataclasses."""

from schemas.configs.algorithm_configs import BaseAlgorithmParams


class TestBaseAlgorithmParams:
    """Test suite for BaseAlgorithmParams."""

    def test_initialization(self):
        """Test initialization with default and custom values."""
        # Default initialization
        params = BaseAlgorithmParams()
        assert params.random_state is None
        assert params.verbose is False

        # Custom initialization
        params = BaseAlgorithmParams(random_state=42, verbose=True)
        assert params.random_state == 42
        assert params.verbose is True


