import numpy as np
import pandas as pd
import pytest
import os
sciezka = os.path.abspath("C:\\Users\\pc60303\\Documents\\ml-playground\\api\\algorithms")
from algorithms.supervised.naive_bayes_classificator import NaiveBernoulliClassifier
from schemas.configs.naive_bayes_config import NaiveBayesParams

@pytest.fixture(scope="module")
def titanic_dataset():
    # Wczytaj dane
    df = pd.read_csv("C:\\Users\\pc60303\\Documents\\ml-playground\\datasets\\titanic.csv")
    # Preprocessing:
    # - wybieramy kilka prostych cech do binarnego zakodowania
    # - zakodujemy je "na sztywno" (tu: płeć, klasa, port wypłynięcia)
    df = df.copy()
    df["Sex_female"] = (df["Sex"] == "female").astype(int)
    df["Sex_male"] = (df["Sex"] == "male").astype(int)
    # Zakoduj klasy (1st, 2nd, 3rd) jako cechy binarne
    for pclass in df["Pclass"].unique():
        df[f"Pclass_{pclass}"] = (df["Pclass"] == pclass).astype(int)
    # Zakoduj port wypłynięcia
    for embark in df["Embarked"].dropna().unique():
        df[f"Embarked_{embark}"] = (df["Embarked"] == embark).astype(int)
    # Wybieramy tylko binarne cechy
    feature_cols = (
        [c for c in df.columns if c.startswith("Sex_")]
        + [c for c in df.columns if c.startswith("Pclass_")]
        + [c for c in df.columns if c.startswith("Embarked_")]
    )
    # Target
    y = df["Survived"].values
    X = df[feature_cols].values
    return X, y

def test_fit_and_predict_on_titanic(titanic_dataset):
    X, y = titanic_dataset
    model = NaiveBernoulliClassifier(NaiveBayesParams(alpha=1.0, verbose=False))
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert np.all(np.isin(preds, [0, 1]))
    # Test: accuracy nie powinna być przypadkowa (dla zabawkowego zbioru np. >0.5)
    accuracy = np.mean(preds == y)
    assert 0.5 <= accuracy <= 1.0

def test_predict_proba_shape_and_sum_on_titanic(titanic_dataset):
    X, y = titanic_dataset
    model = NaiveBernoulliClassifier()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (X.shape[0], 2)
    np.testing.assert_almost_equal(probs.sum(axis=1), np.ones(X.shape[0]))

def test_score_keys_and_ranges_on_titanic(titanic_dataset):
    X, y = titanic_dataset
    model = NaiveBernoulliClassifier()
    model.fit(X, y)
    scores = model.score(X, y)
    assert "accuracy" in scores
    assert "log_loss" in scores
    assert 0 <= scores["accuracy"] <= 1
    assert scores["log_loss"] >= 0

def test_input_validation_raises_on_titanic(titanic_dataset):
    X, _ = titanic_dataset
    y_bad = np.ones(X.shape[0]) * 2  # Złe etykiety
    model = NaiveBernoulliClassifier()
    with pytest.raises(ValueError):
        model.fit(X, y_bad)
    X_bad = X.copy()
    X_bad[0, 0] = 2  # Nie binarna cecha
    y = np.zeros(X.shape[0])
    with pytest.raises(ValueError):
        model.fit(X_bad, y)

def test_get_parameters_result_on_titanic(titanic_dataset):
    X, y = titanic_dataset
    model = NaiveBernoulliClassifier()
    model.fit(X, y)
    params = model.get_parameters()
    assert "class_probs" in params
    assert "feature_probs" in params
    assert "alpha" in params
    assert isinstance(params["class_probs"], np.ndarray)
    assert isinstance(params["feature_probs"], np.ndarray)