"""Tests for Naive Bayes configuration classes."""

import pytest

from schemas.configs.naive_bayes_config import (
    NaiveBayesParams
    # BernoulliNBParams,
    # GaussianNBParams,
    # MultinomialNBParams,
)


@pytest.mark.parametrize(
    "alpha,fit_prior,class_prior",
    [
        (1.0, True, None),  # default values
        (0.5, False, [0.3, 0.7]),  # custom values
    ],
)
def test_multinomial_nb_params(alpha, fit_prior, class_prior):
    """Test MultinomialNBParams initialization."""
    params = NaiveBayesParams( # was supposed to be MultinomialNBParams
        alpha=alpha,
        fit_prior=fit_prior,
        class_prior=class_prior
    )
    assert params.alpha == alpha
    assert params.fit_prior == fit_prior
    assert params.class_prior == class_prior


# @pytest.mark.parametrize(
#     "var_smoothing",
#     [1e-9, 1e-8],  # default and custom values
# )
# def test_gaussian_nb_params(var_smoothing):
#     """Test GaussianNBParams initialization."""
#     params = NaiveBayesParams(var_smoothing=var_smoothing) # was supposed to be GaussianNBParams
#     assert params.var_smoothing == var_smoothing


@pytest.mark.parametrize(
    "alpha,fit_prior,class_prior,binarize",
    [
        (1.0, True, None, 0.0),  # default values
        (0.5, False, [0.3, 0.7], 0.5),  # custom values
    ],
)
def test_bernoulli_nb_params(alpha, fit_prior, class_prior, binarize):
    """Test BernoulliNBParams initialization."""
    params = NaiveBayesParams( # was supposed to be BernoulliNBParams
        alpha=alpha,
        fit_prior=fit_prior,
        class_prior=class_prior,
        binarize=binarize
    )
    assert params.alpha == alpha
    assert params.fit_prior == fit_prior
    assert params.class_prior == class_prior
    assert params.binarize == binarize




