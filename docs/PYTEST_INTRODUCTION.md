# Introduction to Pytest for ML Algorithm Development

## Overview

This document introduces pytest, the testing framework used in our ML Platform. Understanding pytest will help you effectively develop and test new algorithms.

## What is Pytest?

Pytest is a powerful Python testing framework that simplifies test writing with its simple syntax, powerful fixture system, and extensive plugin ecosystem.

Key features:

- Simple syntax: Uses standard Python `assert` statements
- Fixtures: Reusable setup/teardown
- Parameterization: Run the same test with multiple inputs
- Plugins: Extend functionality for specific needs
- Auto-discovery: Automatically finds test files and functions

## Basic Test Structure

A basic pytest test is a Python function with a name starting with `test_`:

```python
# test_example.py
def test_addition():
    result = 2 + 2
    assert result == 4

def test_string_operation():
    result = "hello" + " world"
    assert result == "hello world"
    assert len(result) == 11
```

## Running Tests

In our platform, run tests using Docker:

```bash
# Run FastAPI tests
docker-compose exec fastapi pytest

# Run specific test file
docker-compose exec fastapi pytest tests/test_supervised/test_linear_regression.py

# Run specific test
docker-compose exec fastapi pytest tests/test_supervised/test_linear_regression.py::TestLinearRegression::test_fit_predict
```

## Understanding Fixtures

Fixtures provide reusable setup for tests:

```python
# In conftest.py
import pytest
import numpy as np

@pytest.fixture
def simple_dataset():
    """Create a simple dataset for testing."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([3, 7, 11])
    return X, y

# In test file
def test_algorithm_fit(simple_dataset):
    """Test that algorithm fits correctly."""
    X, y = simple_dataset
    # Use the dataset in the test
```

Our test suite provides several dataset fixtures in `conftest.py`:

- `simple_linear_dataset`: For regression
- `simple_classification_dataset`: For classification
- `simple_clustering_dataset`: For clustering

## Assertions and Testing

Pytest uses standard Python assert statements with enhanced failure messages:

```python
def test_model_prediction(simple_dataset):
    X, y = simple_dataset
    model = YourAlgorithm()
    model.fit(X, y)
    predictions = model.predict(X)

    # Basic assertions
    assert predictions.shape == y.shape
    assert isinstance(predictions, np.ndarray)

    # Numerical assertions (with tolerance)
    assert np.allclose(predictions, y, rtol=0.1, atol=0.1)

    # Testing for exceptions
    with pytest.raises(ValueError):
        model.predict("invalid input")
```

## Test Organization

Organize tests using classes for logical grouping:

```python
class TestYourAlgorithm:
    """Group tests for YourAlgorithm."""

    def test_initialization(self):
        """Test initialization."""
        # Test code

    def test_fit(self, simple_dataset):
        """Test fitting."""
        # Test code
```

## Parameterized Tests

Run the same test with different inputs:

```python
@pytest.mark.parametrize("learning_rate,expected_performance", [
    (0.01, 0.8),
    (0.1, 0.85),
    (0.001, 0.75)
])
def test_learning_rate_impact(simple_dataset, learning_rate, expected_performance):
    """Test impact of different learning rates."""
    X, y = simple_dataset
    model = YourAlgorithm(params=YourAlgorithmParams(learning_rate=learning_rate))
    model.fit(X, y)
    score = model.score(X, y)
    assert score >= expected_performance
```

## Test Coverage

Assess test coverage with pytest-cov:

```bash
docker-compose exec fastapi pytest --cov=algorithms
```

## Advanced Features

1. Markers: Tag tests for selective running

   ```python
   @pytest.mark.slow
   def test_long_running():
       # Long-running test
   ```

2. Fixture Scopes: Control setup/teardown frequency

   ```python
   @pytest.fixture(scope="module")
   def expensive_setup():
       # This runs once per module
   ```

3. Mocking: Replace complex dependencies
   ```python
   def test_with_mock(mocker):
       mocker.patch('module.function', return_value=42)
   ```

## Best Practices

1. Write Independent Tests: Tests should not depend on each other
2. Keep Tests Fast: Fast tests encourage frequent running
3. Use Descriptive Names: Make test names descriptive
4. One Assert Per Test: Focus each test on one behavior
5. Test Expected and Unexpected Paths: Test both valid and invalid inputs

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Parameterizing Tests](https://docs.pytest.org/en/stable/parametrize.html)
- [Assertion Introspection](https://docs.pytest.org/en/stable/assert.html)
