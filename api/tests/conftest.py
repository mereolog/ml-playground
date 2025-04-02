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


@pytest.fixture
def simple_linear_dataset():
    """Create a simple linear dataset for regression."""
    np.random.seed(42)
    X = np.random.rand(100, 3)
    true_weights = np.array([1.5, -0.8, 2.0])
    true_bias = 0.5
    y = np.dot(X, true_weights) + true_bias + np.random.normal(0, 0.1, size=100)
    return X, y, true_weights, true_bias


@pytest.fixture
def simple_classification_dataset():
    """Create a simple dataset for binary classificationg."""
    pass


@pytest.fixture
def simple_clustering_dataset():
    """Create a simple dataset for clustering."""
    pass
