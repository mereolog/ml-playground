from abc import abstractmethod
from typing import Generic, TypeVar

from algorithms.base.algorithm import Algorithm
from schemas.configs.algorithm_configs import UnsupervisedAlgorithmsParams

SP = TypeVar("SP", bound=UnsupervisedAlgorithmsParams)


class UnsupervisedAlgorithm(Algorithm[SP], Generic[SP]):
    """
    Base class for unsupervised algorithms
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def params(self) -> SP:
        """Unsupervised algorithm params"""
        pass
