# pytest tests/test_supervised/test_logistic_regression_dataset.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from algorithms.supervised.logistic_regression import LogisticRegression
from schemas.configs.logistic_regression import LogisticRegressionParams


def test_logistic_regression_on_social_network_ads():
    df = pd.read_csv("datasets/Social_Network_Ads.csv")
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])

    X = df[["Gender", "Age", "EstimatedSalary"]].values
    y = df["Purchased"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    params = LogisticRegressionParams(
        learning_rate=0.05,  
        epochs=500,
        batch_size=32,
        threshold=0.5,
    )

    model = LogisticRegression(params)
    model.fit(X, y)

    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y)

    assert accuracy > 0.6, f"Expected accuracy > 0.6 but got {accuracy:.4f}"

