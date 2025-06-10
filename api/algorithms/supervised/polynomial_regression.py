"""
This module contains implementation of the Polynomial Regression algorithm.
Polynomial Regression is a form of regression analysis in which the relationship between
independent variables and the dependent variable is modeled as an nth-degree polynomial.
"""

import logging
from typing import Dict, List, Optional, Any

import numpy as np

from algorithms.supervised.linear_regression import LinearRegression
from schemas.configs.polynomial_regression import PolynomialRegressionParams


class PolynomialRegression(LinearRegression):
    """
    Polynomial Regression implementation using gradient descent.
    Extends Linear Regression by transforming input features into polynomial features before fitting.
    """

    def __init__(self, params: Optional[PolynomialRegressionParams] = None):
        """
        Initialize the Polynomial Regression model.
        Args:
            params: Configuration parameters for the model. Uses defaults if None.
        """
        super().__init__(params=params)

        self._params: PolynomialRegressionParams = (
            params if params is not None else PolynomialRegressionParams()
        )

        self.degree = self._params.degree
        self.include_bias = self._params.include_bias

        self._mean = None
        self._std = None

        self.logger = logging.getLogger(__name__)
        if self.params.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

    def _create_polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform input features into polynomial features up to a given degree.
        Also applies normalization (zero mean, unit variance).
        Args:
            X: Input features, shape (n_samples, n_features).
        Returns:
            Transformed feature matrix with polynomial terms.
        """
        n_samples, n_features = X.shape
        poly_features = []

        for sample in X:
            features = []
            if self.include_bias:
                features.append(1.0)  # Bias term (intercept)

            for d in range(1, self.degree + 1):
                for feat in sample:
                    features.append(feat ** d)

            poly_features.append(features)

        poly_features = np.array(poly_features)

        # Normalization
        if self._mean is None or self._std is None:
            self._mean = np.mean(poly_features, axis=0)
            self._std = np.std(poly_features, axis=0)
            self._std[self._std == 0] = 1  # Avoid division by zero

        return (poly_features - self._mean) / self._std

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialRegression":
        """
        Fit the polynomial regression model to the training data.
        Transforms input features into polynomial features before calling linear regression's fit.
        Args:
            X: Training data features, shape (n_samples, n_features).
            y: Target values, shape (n_samples,).
        Returns:
            Self reference for method chaining.
        """
        X_poly = self._create_polynomial_features(X)
        return super().fit(X_poly, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values using the trained polynomial model.
        Transforms input features before calling linear regression's predict.
        Args:
            X: Input features, shape (n_samples, n_features).
        Returns:
            Predicted values, shape (n_samples,).
        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self._mean is None or self._std is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        X_poly = self._create_polynomial_features(X)
        return super().predict(X_poly)

    def get_coefficients(self) -> Dict[str, Any]:
        """
        Get model coefficients including polynomial degree.
        Returns:
            Dictionary containing 'weights', 'bias', and 'degree'.
        """
        coeffs = super().get_coefficients()
        coeffs["degree"] = self.degree
        return coeffs

    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history including loss per epoch.
        Returns:
            Dictionary with 'training_loss' key and list of losses.
        """
        return super().get_training_history()
