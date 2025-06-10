#pytest tests/test_supervised/test_logistic_regression_categorical.py -v
import numpy as np
import pandas as pd
import pytest

from algorithms.supervised.logistic_regression import LogisticRegression
from schemas.configs.logistic_regression import LogisticRegressionParams


def test_logistic_regression_with_categorical_data():
    df = pd.DataFrame({
        "Gender": ["Male", "Female", "Female", "Male", "Female"],
        "Age": [22, 35, 58, 44, 29],
        "Purchased": [0, 1, 1, 0, 1]  
    })

    X = pd.get_dummies(df[["Gender", "Age"]], drop_first=True)
    y = df["Purchased"].values

    assert X.shape[0] == y.shape[0]

    params = LogisticRegressionParams(
        learning_rate=0.1,
        epochs=100,
        batch_size=None,
        threshold=0.5,
        regularization="l2",
        lambda_=0.01
    )
    model = LogisticRegression(params)
    model.fit(X.values, y)

    y_pred = model.predict(X.values)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y.shape
    assert set(np.unique(y_pred)).issubset({0, 1})
