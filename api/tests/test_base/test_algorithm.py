"""Tests for the base Algorithm class."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeVar

import pytest
from algorithms.base.algorithm import Algorithm
from schemas.configs.algorithm_configs import BaseAlgorithmParams


@dataclass
class MockParams(BaseAlgorithmParams):
    """Mock parameter class for testing.
    
    This class extends BaseAlgorithmParams to provide a concrete implementation
    for testing purposes with two simple integer parameters.
    
    Attributes:
        param1: First test parameter
        param2: Second test parameter
    """
    param1: int = 1
    param2: int = 2


P = TypeVar("P", bound=MockParams)


class ConcreteAlgorithm(Algorithm[MockParams]):
    """Concrete implementation of Algorithm for testing."""

    def __init__(self):
        self._params = MockParams()

    @property
    def params(self) -> MockParams:
        """Implement the abstract params property."""
        return self._params


class TestBaseAlgorithm:
    """Test suite for the base Algorithm class."""

    def test_concrete_implementation(self):
        """Test that a concrete implementation can be instantiated."""
        model = ConcreteAlgorithm()
        assert isinstance(model, Algorithm)

    def test_get_params(self):
        """Test the get_params method."""
        model = ConcreteAlgorithm()
        params = model.get_params()
        assert params["param1"] == 1
        assert params["param2"] == 2

    def test_set_params(self):
        """Test the set_params method."""
        model = ConcreteAlgorithm()
        model.set_params(param1=10, param2="new")
        assert model.params.param1 == 10
        assert model.params.param2 == "new"
