import numpy as np
from algorithms.supervised.logistic_regression import LogisticRegression
from schemas.configs.logistic_regression import LogisticRegressionParams

def test_logistic_regression_on_simple_data():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])

    params = LogisticRegressionParams(learning_rate=0.1, epochs=100, verbose=False)
    model = LogisticRegression(params=params)

    model.fit(X, y)

    predictions = model.predict(X)

    assert np.array_equal(predictions, y), "the model failed to partition simple data"