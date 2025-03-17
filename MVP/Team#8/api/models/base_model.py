# fastapi/models/base_model.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class MLAlgorithm(ABC):
    """Base interface for all ML algorithms."""

    @abstractmethod
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the model with parameters.

        Args:
            params: Dictionary of algorithm parameters
        """
        self.params = params or {}
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model on provided data.

        Args:
            X: Training features
            y: Target values (optional for unsupervised algorithms)

        Returns:
            Dictionary containing training metrics and history
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Features to predict on

        Returns:
            Model predictions
        """
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True target values

        Returns:
            Dictionary of evaluation metrics
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        return self.params

    @abstractmethod
    def set_params(self, params: Dict[str, Any]) -> None:
        """
        Set model parameters.

        Args:
            params: Dictionary of model parameters
        """
        self.params.update(params)


class SupervisedAlgorithm(MLAlgorithm):
    """Base interface for supervised learning algorithms."""

    pass


class UnsupervisedAlgorithm(MLAlgorithm):
    """Base interface for unsupervised learning algorithms."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: None = None) -> Dict[str, Any]:
        pass
