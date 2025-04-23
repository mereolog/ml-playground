import numpy as np
import pytest
from utils.losses import LogLoss
def test_log_loss_binary():
    # Dane binarne
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.6, 0.4])

    # Obliczanie straty Log Loss
    loss = LogLoss.compute(y_true, y_pred)

    # Oczekiwany wynik (obliczony ręcznie lub za pomocą innego narzędzia)
    expected_loss = -np.mean(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )

    # Sprawdzenie poprawności wyniku
    assert np.isclose(loss, expected_loss), f"Expected {expected_loss}, got {loss}"

def test_log_loss_multiclass():
    # Dane wieloklasowe (one-hot encoded)
    y_true = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    y_pred = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
    ])

    # Obliczanie straty Log Loss
    loss = LogLoss.compute(y_true, y_pred)

    # Oczekiwany wynik (obliczony ręcznie lub za pomocą innego narzędzia)
    expected_loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    # Sprawdzenie poprawności wyniku
    assert np.isclose(loss, expected_loss), f"Expected {expected_loss}, got {loss}"

def test_log_loss_gradient_binary():
    # Dane binarne
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8, 0.3, 0.7])

    # Obliczanie gradientu
    gradient = LogLoss.gradient(y_true, y_pred)

    # Oczekiwany gradient (obliczony ręcznie)
    expected_gradient = -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)

    # Sprawdzenie poprawności wyniku
    assert np.allclose(gradient, expected_gradient), f"Expected {expected_gradient}, got {gradient}"

def test_log_loss_gradient_multiclass():
    # Dane wieloklasowe (one-hot encoded)
    y_true = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    y_pred = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.2, 0.1, 0.7],
    ])

    # Obliczanie gradientu
    gradient = LogLoss.gradient(y_true, y_pred)

    # Oczekiwany gradient (obliczony ręcznie)
    expected_gradient = -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)

    # Sprawdzenie poprawności wyniku
    assert np.allclose(gradient, expected_gradient), f"Expected {expected_gradient}, got {gradient}"

def test_log_loss_clipping():
    # Testowanie przybliżeń dla wartości bliskich 0 i 1
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0.999999, 0.000001, 0.5])

    # Obliczanie straty Log Loss
    loss = LogLoss.compute(y_true, y_pred)

    # Sprawdzenie, czy funkcja nie zwraca błędu (np. log(0))
    assert np.isfinite(loss), "Log Loss returned an invalid value (infinite or NaN)"