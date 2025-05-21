from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any, Generic
from typing import TypeVar

P = TypeVar("P") 

class SupervisedAlgorithm(ABC, Generic[P]):

    def __init__(self):
        self._params = None  

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

    