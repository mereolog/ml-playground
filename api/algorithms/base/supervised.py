from abc import abstractmethod
from typing import Generic, TypeVar, Dict, Any, List
import numpy as np

from algorithms.base.algorithm import Algorithm, P
from schemas.configs.algorithm_configs import SupervisedAlgorithmsParams

SP = TypeVar("SP", bound=SupervisedAlgorithmsParams)


class SupervisedAlgorithm(Algorithm[SP], Generic[SP]):
    """
    Base class for supervised algorithms
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def params(self) -> P:
        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        pass

    @abstractmethod
    def get_coefficients(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_training_history(self) -> Dict[str, List[float]]:
        pass

