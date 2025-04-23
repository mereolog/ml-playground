"""Tests for Naive Bayes configuration dataclasses."""

import pytest
from schemas.configs.naive_bayes_configs import NaiveBayesParams


class TestNaiveBayesParams:
    """Test suite for NaiveBayesParams."""

    def test_default_initialization(self):
        """Test initialization with default values."""
        params = NaiveBayesParams()
        assert params.alpha == 1.0

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        params = NaiveBayesParams(
            alpha=0.5,
        )
        assert params.alpha == 0.5

    def test_invalid_alpha(self):
        """Test validation of negative alpha parameter."""
        with pytest.raises(ValueError, match="alpha must be greater than or equal to 0"):
            NaiveBayesParams(alpha=-1.0)

    def test_multiple_invalid_params(self):
        """Test multiple invalid parameters together."""
        # this is how we can test multiple invalid parameters together
        with pytest.raises(ValueError, match="alpha must be greater than or equal to 0"):
            NaiveBayesParams(alpha=-1.0, binarize=-0.5)
        
        with pytest.raises(ValueError, match="alpha must be greater than or equal to 0"):
            NaiveBayesParams(alpha=-2.0, binarize=-1.0, fit_prior=False)


        
            
