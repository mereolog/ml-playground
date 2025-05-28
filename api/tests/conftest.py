"""Common test fixtures and utilities for all algorithm tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add the project root directory to the Python path
# This works regardless of where the container's working directory is set

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# So basically what we are doing here is instead of using a
# specific dataset for testing (which btw we could do) we will
# be generating synthetic data and passing it to the test using
# something called "fixtures"


@pytest.fixture(scope="function")
def simple_linear_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Generate a simple linear dataset for testing."""
    X = np.array([[i] for i in range(0, 10)], dtype=np.float32)
    y = 2 * X + 1  # y = 2x + 1
    return X, y.flatten()


@pytest.fixture(scope="function")
def simple_polynomial_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Generate a simple polynomial dataset for testing."""
    X = np.array([[i] for i in range(0, 10)], dtype=np.float32)
    y = 2 * X**2 + 3 * X + 1  # y = 2x^2 + 3x + 1
    return X, y.flatten()
