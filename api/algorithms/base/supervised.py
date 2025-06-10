from abc import abstractmethod
from typing import Generic, TypeVar

from algorithms.base.algorithm import Algorithm
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
    def params(self) -> SP:
        """Supervised algorithm params"""
        pass
