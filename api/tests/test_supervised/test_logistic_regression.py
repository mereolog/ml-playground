import numpy as np
from schemas.configs.logistic_regression import LogisticRegressionParams
from utils.metrics import accuracy_score
from api.supervised.logistic_regression import LogisticRegression 

def test_logistic_regression_on_simple_data():
    X = np.array([["red"], ["green"], ["blue"], ["red"]])
    y = np.array([0, 0, 1, 1])

    params = LogisticRegressionParams(learning_rate=0.1, epochs=1000, verbose=False)
    model = LogisticRegression(params=params)

    model.fit(X, y)
    predictions = model.predict(X)
    
    accuracy = accuracy_score(y, predictions)
    assert accuracy >= 0.75, f"Expected accuracy >= 0.75 but got {accuracy}"

