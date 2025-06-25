# pytest tests/test_supervised/test_knn_car.py -v
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from algorithms.supervised.knn import KNearestNeighbor
from schemas.configs.k_nearest_neighbour_algorithm import KNeighborsParams


def test_knn_on_car_dekho_dataset():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    csv_path = os.path.join(project_root, "datasets", "CAR DETAILS FROM CAR DEKHO.csv")

    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)

    threshold_price = df["selling_price"].median()
    df["target"] = (df["selling_price"] >= threshold_price).astype(int)

    for col in ["fuel", "seller_type", "transmission", "owner"]:
        df[col] = LabelEncoder().fit_transform(df[col])

    features = ["year", "km_driven", "fuel", "seller_type", "transmission", "owner"]
    X = df[features].values
    y = df["target"].values

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = KNeighborsParams(
        test_size=0.2,
        validation_size=None,
        shuffle=True,
        stratify=False,
        n_neighbors=5,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski"
    )
    model = KNearestNeighbor(params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    assert accuracy > 0.6, f"KNN accuracy too low: {accuracy:.4f}"
