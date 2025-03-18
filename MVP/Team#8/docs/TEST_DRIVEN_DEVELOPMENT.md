# Test-Driven Development guide for developing ml algorithms

## What is Test-Driven Development?

Test-Driven Development (TDD) is a software development approach where tests are written before implementing the actual code. This reversal of the traditional development process provides several benefits:

1. Clear requirements: Tests clearly tell how your code should behave
2. Focused development: Only implement what's needed to pass the tests
3. Built-in verification: Ensure code works as expected from the start
4. Regression protection: Tests catch breaking changes immediately
5. Better design: TDD naturally leads to more modular, loosely

The TDD cycle follows three simple steps, often called **"Red-Green-Refactor"**:

1. Red: Write a failing test for the functionality you want to implement, it is supposed to fail
2. Green: Write the simplest code that makes the test pass
3. Refactor: Improve the code while ensuring tests continue to pass

## Prerequisites

Before starting, ensure you have:

- Docker running the application
- Understanding of the foundations of the algorithm you want to implement
- Something, I can't remember right now

## Running tests in our Docker container 
This will allow us to run pytest inside of a FastAPI container
```bash
docker compose up --build -d
docker ps
# command output
CONTAINER ID   IMAGE           COMMAND                  CREATED         STATUS         PORTS         
           NAMES
bb2df3e7ce8e   team8-fastapi   "uvicorn main:app --…"   7 seconds ago   Up 6 seconds   0.0.0.0:8000->
8000/tcp   fastapi-application
cdd7cc75816d   team8-django    "bash /app/entrypoin…"   7 seconds ago   Up 6 seconds   0.0.0.0:8050->
8050/tcp   django
bd9bf27c7f9d   team8-deno      "/tini -- docker-ent…"   7 seconds ago   Up 6 seconds                 
           deno
```

Our fastapi container is accessible under the **fastapi-application** name. Pytest will run all python scripts that have "test_" prefix.

```bash
docker exec -it fastapi-application pytest -v

api % docker exec -it fastapi-application pytest -v 
======================================== test session starts ========================================
platform linux -- Python 3.11.4, pytest-8.3.5, pluggy-1.5.0 -- /usr/local/bin/python
cachedir: .pytest_cache
rootdir: /api
plugins: anyio-4.9.0
collected 11 items                                                                                  

tests/test_configs/test_algorithm_configs.py::TestBaseAlgorithmParams::test_initialization PASSED [  
9%]
tests/test_configs/test_algorithm_configs.py::TestLinearRegressionParams::test_inheritance PASSED [ 1
8%]
tests/test_configs/test_algorithm_configs.py::TestLinearRegressionParams::test_initialization PASSED 
[ 27%]
tests/test_supervised/test_linear_regression.py::TestLinearRegression::test_initialization PASSED [ 3
6%]
tests/test_supervised/test_linear_regression.py::TestLinearRegression::test_get_set_params PASSED [ 4
5%]
tests/test_supervised/test_linear_regression.py::TestLinearRegression::test_fit_predict PASSED [ 54%]
tests/test_supervised/test_linear_regression.py::TestLinearRegression::test_score PASSED      [ 63%]
tests/test_supervised/test_linear_regression.py::TestLinearRegression::test_batch_training PASSED [ 7
2%]
tests/test_supervised/test_linear_regression.py::TestLinearRegression::test_regularization PASSED [ 8
1%]
tests/test_supervised/test_linear_regression.py::TestLinearRegression::test_get_coefficients PASSED [
 90%]
tests/test_supervised/test_linear_regression.py::TestLinearRegression::test_untrained_errors PASSED [
100%]

======================================== 11 passed in 0.08s =========================================
```
## PyTest Flags
When we run pytest with the `-v` flag, we are enabling verbose output. The percentage values you're seeing on the left side are a progress indicator showing how much of the test suite has been completed. 
Here's an explanation of common pytest flags and output elements:

### Some most common pytest flags:

- `-v` or `--verbose`: Increases verbosity. Shows more detailed test results.
- `-q` or `--quiet`: Decreases verbosity.
- `-s`: Shows print statements from your tests (normally we woulnd't see any prints in the console).
- `-x` or `--exitfirst`: Stops after the first failure.
- `-m "mark"`: Runs tests with specific markers (e.g., `-m "slow"`)

### Progress Output:

When you run pytest, here's what the output elements mean:

- `[100%]` - The percentage of tests completed
- `.` - A test passed
- `F` - A test failed
- `E` - An error occurred during setup or teardown
- `s` - A test was skipped
- `x` - A test was xfailed (expected to fail)
- `X` - A test was xpassed (passed but was expected to fail)

### Example output:

```bash
collecting ... collected 42 items

test_file.py::test_function1 PASSED                                     [  2%]
test_file.py::test_function2 FAILED                                     [  5%]
```
## The TDD Cycle for ML Algorithm Development

### 1. Identify Algorithm Requirements

Start by defining what your algorithm needs to do:

- What are the expected inputs and outputs?
- What configuration parameters are required?
- What mathematical properties should be preserved?
- What edge cases need to be handled?

### 2. Write a Failing Test

Start with a test that defines the expected behavior.
In this case we are initializing an algorithm object and checking if it is correcly initialized with default parameters
```python
# /api/tests/test_supervised/test_xyz_algorithm.py
def test_initialization():
    """Test that the model initializes with correct parameters."""
    model = XyzAlgorithm()
    
    # some, but not all assertions deserve to have a message that is going to be printed when asserion fails
    # in this case the assertions ale self explanatory but you get where I am going with this

    assert model.params.learning_rate == 0.01
    assert model.params.max_iterations == 100, "Default model max_iterations parameter should be set to 100"
```

Now we will learn something new. Lets run this one specific test that we just wrote.

```bash
# running a test function
docker exec -it fastapi-application pytest tests/test_supervised/test_xyz_algorithm.py::test_initialization -v

# if we were to run a test method of a specific class
docker exec -it fastapi-application pytest tests/test_supervised/test_xyz_algorithm.py::TestXYZ::test_initialization -v
```
Everything fails, but at least we can start somewhere.
### 3. Implement Just Enough Code

Create the minimal implementation to make the test pass:

```python
@dataclass
class XyzAlgorithmParams(BaseAlgorithmParams):
    learning_rate: float = 0.01
    max_iterations: int = 100
```

And the basic class structure:

```python
class XyzAlgorithm(Algorithm):
    def __init__(self, params=None):
        self.params = params if params is not None else XyzAlgorithmParams()
```

Run the test again to see it pass (the Green phase).

### 4. Refactor if Needed

Improve your code without changing functionality:
- Add types
- Remove duplication
- Improve naming
- Enhance structure
- Optimize performance

Verify tests still pass after refactoring.

### 5. Repeat the Cycle

Continue adding tests for new functionality, one small piece at a time:

1. Core algorithm behavior
2. Parameter validation
3. Edge case handling
4. Interface compatibility
5. Performance characteristics

## Effective TDD Practices for ML Algorithms

### Testing Algorithm Configuration

Test that your algorithm accepts and validates parameters correctly. Lets check if the validation correctly raises ValueError when we initialize it with invalid parameter.

```python
def test_invalid_parameters():
    """Test that invalid parameters raise appropriate errors."""
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        XyzAlgorithm(params=XyzAlgorithmParams(learning_rate=-0.1))
```
This test assumes that the **XyzAlgorithmParams** has validation that could look something like this:
```python
@dataclass
class XyzAlgorithmParams(BaseAlgorithmParams):
    learning_rate: float = 0.01
    max_iterations: int = 100

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
```


### Testing Core Mathematical Properties

Verify that your algorithm follows established mathematical principles:

```python
def test_optimization_descent(simple_dataset):
    """Test that loss decreases during training."""
    model = NewAlgorithm()
    model.fit(simple_dataset)

    # Loss should decrease monotonically
    assert all(model.loss_history[i] >= model.loss_history[i+1]
               for i in range(len(model.loss_history)-1))
```

### Testing with Synthetic Data

Create test fixtures with known ground truth. This one for example creates some synthetic data for regression algorithms.

```python
# /api/tests/conftest.py
@pytest.fixture
def simple_linear_dataset():
    """Create dataset with known parameters for testing."""
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    y = 3 * X.ravel() + 2 + np.random.normal(0, 0.5, 100)
    return X, y
```

### Testing Edge Cases

Ensure your algorithm handles extreme or unusual cases. We could handle a situation where a dataset is empty.

```python
def test_empty_dataset():
    """Test behavior with empty input."""
    model = NewAlgorithm()
    with pytest.raises(ValueError, match="Empty dataset"):
        model.fit(np.array([]), np.array([]))
```

## Using the Test Suite for New Algorithms

### Directory Structure

Place your test files in the appropriate directory:

```
tests/
├── test_supervised/       # For supervised learning algorithms
│   └── test_new_algo.py   # Your new algorithm tests
├── test_unsupervised/     # For unsupervised learning algorithms
└── conftest.py            # Shared test fixtures
```

### Running Specific Tests
In previous examples we were executing the tests using the docker exec command.
It is convenient to access the container shell to execute test commands as if it was happening on our host system.

To do that we need to run this command:
```bash
user@user % docker exec -it fastapi-application bash
root@8db2258f4c2f:/api# 
```
Now we can run the tests using pytest.
```bash
# Run all tests for your algorithm
pytest tests/test_supervised/test_new_algo.py -v

# Run a specific test
pytest tests/test_supervised/test_new_algo.py::TestNewAlgo::test_fit -v
```

### Testing Integration Points

Ensure your algorithm integrates with the rest of the platform:

```python
def test_inheritance_and_interfaces():
    """Test that algorithm follows required interfaces."""
    model = NewAlgorithm()
    assert isinstance(model, SupervisedAlgorithm)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    assert hasattr(model, "get_params")
```

## Example TDD Workflow for a New Algorithm

### 1. Start With Configuration Testing

1. Write test for algorithm parameters
2. Implement parameter class
3. Run test to verify

### 2. Add Initialization Testing

1. Write test for algorithm initialization
2. Implement basic class structure
3. Run test to verify

### 3. Add Core Algorithm Behavior

1. Write test for fitting algorithm to data
2. Implement basic fitting functionality
3. Run test to verify

### 4. Add Prediction Testing

1. Write test for prediction functionality
2. Implement prediction method
3. Run test to verify

### 5. Add Validation and Edge Cases

1. Write tests for parameter validation
2. Write tests for edge cases
3. Implement validation logic
4. Run tests to verify

### 6. Add Performance Testing

1. Write test for expected performance metrics
2. Run tests to verify the algorithm meets requirements

## Conclusion

Test-Driven Development provides a structured approach to implementing machine learning algorithms that ensures correctness, maintainability, and integration with the existing platform. By writing tests first and following the Red-Green-Refactor cycle, you can confidently build robust implementations that meet all requirements.

Remember that TDD is iterative and incremental - build your algorithm one small piece at a time, always guided by tests. This approach leads to better design decisions and more reliable code.