from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Dict, Generic, TypeVar

from schemas.configs.algorithm_configs import BaseAlgorithmParams

P = TypeVar("P", bound=BaseAlgorithmParams)


class Algorithm(ABC, Generic[P]):
    """Base class for all machine learning algorithms."""

    @property
    @abstractmethod
    def params(self) -> P:
        """
        Algorithm parameters. Must be implemented by subclasses.
        Returns an instance of a BaseAlgorithmParams subclass.
        """
        pass

    def get_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the algorithm.

        Returns:
            Dictionary containing the algorithm parameters
        """
        return self.params.model_dump()

    def set_params(self, **params) -> "Algorithm":
        """
        Set the parameters of the algorithm.

        Args:
            **params: Keyword arguments representing parameter names and values

        Returns:
            Self reference for method chaining
        """
        for key, value in params.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")

        return self
