# HOW_TO.md: Developing ML Algorithms for the Platform

This guide provides step-by-step instructions for developing new machine learning algorithms for our ML Platform. It's designed specifically for team members who will be working on implementing algorithms within the FastAPI service.

## Contents

1. [Getting Started](#getting-started)
2. [Development Environment](#development-environment)
3. [Adding a New ML Algorithm](#adding-a-new-ml-algorithm)
4. [Test-Driven Development Process](#test-driven-development-process)
5. [Common Issues and Solutions](#common-issues-and-solutions)
6. [Example: Implementing a New Algorithm](#example-implementing-a-new-algorithm)

## Getting Started

As an ML algorithm developer, you'll primarily work with the **FastAPI service** located in the `api/` directory.

## Development Environment

### Setup

1. **Clone the repository and start containers**:

   ```bash
   docker-compose up --build
   docker ps
   ```

2. **Access the FastAPI container for development**:

   ```bash
   docker exec -it fastapi-application bash
   ```

3. **Run tests** to verify everything is working:

   ```bash
   pytest tests/ -v
   ```

## Adding a New ML Algorithm

Follow these steps to implement a new algorithm:

### 1. Identify the Algorithm Type

Determine whether your algorithm is:

- **Supervised** (requires labeled data)
- **Unsupervised** (works with unlabeled data)

### 2. Create Model Files

1. **Create a new Python file** in the appropriate directory:

   ```
   # For supervised algorithms:
   api/algorithms/supervised/your_algorithm.py

   # For unsupervised algorithms:
   api/algorithms/unsupervised/your_algorithm.py
   ```

2. **Implement the algorithm class**, extending the appropriate base class:

   ```python
   from algorithms.base.supervised import SupervisedAlgorithm

   class YourAlgorithm(SupervisedAlgorithm):
       """Your algorithm description."""

       def __init__(self, params=None):
           super().__init__(params)
           # Initialize algorithm-specific attributes

       # Rest of implementation
   ```

### 3. Create Configuration Class

1. **Add parameter configurations** in `api/schemas/configs/algorithms_configs.py`:

   ```python
   @dataclass
   class YourAlgorithmParams(BaseAlgorithmParams):
       learning_rate: float = 0.01
       max_iterations: int = 100
       # Other parameters specific to your algorithm

       def __post_init__(self):
           """Validate parameters"""
           if self.learning_rate <= 0:
               raise ValueError("learning_rate must be positive")
           # Rest of validation
   ```

### 4. Implement Core Methods

Your algorithm must implement these **required methods**:

```python
def fit(self, X, y=None):
    """Train the model on data."""
    # Implementation logic for training
    self.is_fitted = True
    return {"loss_history": self.loss_history}

# Rest of required methods (predict, evaluate, etc.)
```

## Test-Driven Development Process

We follow a **TDD approach** for algorithm development:

### 1. Write Tests First

Create a test file in the appropriate directory:

```
# For supervised algorithms:
api/tests/test_supervised/test_your_algorithm.py

# For unsupervised algorithms:
api/tests/test_unsupervised/test_your_algorithm.py
```

### 2. Follow the "Red-Green-Refactor" Cycle

1. **Red**: Write a failing test that defines the expected behavior:

   ```python
   def test_initialization():
       """Test that the model initializes with correct parameters."""
       model = YourAlgorithm()
       assert model.params.learning_rate == 0.01
       assert model.params.max_iterations == 100
   ```

2. **Green**: Implement the simplest code that makes the test pass
3. **Refactor**: Improve the code while ensuring tests still pass

### 3. Test Different Aspects

According to our personal knowledge base on **Effective TDD Practices for ML Algorithms**, you should write tests for:

- **Initialization**: Parameter validation
- **Core functionality**: Training, prediction
- **Edge cases**: Empty datasets, invalid inputs
- **Mathematical properties**: Convergence, expected behavior

For example, testing mathematical properties:

```python
def test_optimization_descent(simple_dataset):
    """Test that loss decreases during training."""
    model = NewAlgorithm()
    model.fit(simple_dataset)
    # Loss should decrease monotonically
    assert all(model.loss_history[i] >= model.loss_history[i+1]
              for i in range(len(model.loss_history)-1))
```

### 4. Run Tests

```bash
# Run all tests for your algorithm
pytest tests/test_supervised/test_your_algorithm.py -v

# Run a specific test
pytest tests/test_supervised/test_your_algorithm.py::test_initialization -v
```

## Common Issues and Solutions

### Issue 1: Integration with Web Interface

Your algorithm will be accessed via WebSockets from the Django frontend. Ensure your code:

- **Returns JSON-serializable objects** from all methods
- **Handles incremental updates** during training for real-time visualization
- **Properly validates input data**

### Issue 2: Performance Considerations

- **Optimize computation-heavy sections** using NumPy
- **Consider batch processing** for large datasets
- **Return incremental results** rather than waiting for completion

### Issue 3: Testing Synthetic Data

Use the provided test fixtures in `api/tests/conftest.py` for consistent testing:

```python
def test_with_simple_dataset(simple_dataset):
    """Test algorithm with standard test dataset."""
    X, y = simple_dataset
    model = YourAlgorithm()
    results = model.fit(X, y)
    # Assertions
```

## Example: Implementing a New Algorithm

Here's a step-by-step example of implementing a new algorithm (K-means clustering):

### 1. Create Test File First

```python
# THIS CODE WAS NOT TESTED AND ACTS JUST AS AN EXAMPLE, LOOK AT ACTUAL IMPLEMENTATION FOR REFERENCE

# tests/test_unsupervised/test_kmeans.py
import pytest
import numpy as np
from algorithms.unsupervised.kmeans import KMeansAlgorithm, KMeansParams

def test_initialization():
    """Test KMeans initialization."""
    model = KMeansAlgorithm()
    assert model.params.n_clusters == 3
    assert model.params.max_iterations == 100

def test_fit_predict(synthetic_clusters):
    """Test KMeans clustering on synthetic data."""
    X = synthetic_clusters
    model = KMeansAlgorithm(params=KMeansParams(n_clusters=3))
    results = model.fit(X)

    assert hasattr(model, 'cluster_centers_')
    assert len(model.cluster_centers_) == 3

    labels = model.predict(X)
    assert labels.shape[0] == X.shape[0]
    assert len(np.unique(labels)) <= 3

```

### 2. Implement the Algorithm

```python
# algorithms/unsupervised/kmeans.py
from dataclasses import dataclass
import numpy as np
from algorithms.base.unsupervised import UnsupervisedAlgorithm

class KMeansAlgorithm(UnsupervisedAlgorithm):
    """K-means clustering algorithm."""
    pass
```

### 3. Update Config Schemas

```python
# schemas/configs/algorithms_configs.py
# add your new algorithm parameters
@dataclass
class KMeansParams(BaseAlgorithmParams):
    n_clusters: int = 3
    max_iterations: int = 100
    tolerance: float = 1e-4

    def __post_init__(self):
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
```

### 4. Run Tests to Verify

```bash
pytest tests/test_unsupervised/test_kmeans.py -v
```

By following this example and the TDD process, you'll build robust, well-tested algorithms that integrate seamlessly with our platform.

Remember that **small, incremental changes** with thorough testing lead to the most reliable implementations. Reach out to the team if you encounter any issues!
