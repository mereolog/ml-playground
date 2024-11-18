import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def load_data(file):
    df = pd.read_csv(file)
    return df[['YearsExperience']], df['Salary']

def get_model(reg_type, reg_param):
    if reg_type == "Ridge":
        return Ridge(alpha=reg_param)
    elif reg_type == "Lasso":
        return Lasso(alpha=reg_param)
    return LinearRegression()

def compute_cost(y_true, y_pred, cost_function):
    if cost_function == "MSE":
        return mean_squared_error(y_true, y_pred)
    elif cost_function == "MAE":
        return mean_absolute_error(y_true, y_pred)
    elif cost_function == "R2":
        return r2_score(y_true, y_pred)
    return None

def train_model(X, y, reg_type, reg_param, epochs, cost_function, step_by_step, learning_rate):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = get_model(reg_type, reg_param)
    errors = []

    for epoch in range(epochs):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        error = compute_cost(y_test, y_pred, cost_function)
        errors.append(error)


        if step_by_step:
            yield epoch, error, y_test, y_pred
        else:
            continue

    return model, errors, y_test, y_pred

def plot_errors(errors):
    plt.figure(figsize=(10, 5))
    plt.plot(errors, label="Błąd")
    plt.xlabel("Epoka")
    plt.ylabel("Znaczenie błędu")
    plt.title("Zmiana błędu epoka po epoce")
    plt.legend()
    return plt

def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Rzeczywiste wynagrodzenie", color="blue")
    plt.plot(y_pred, label="Przewidywane wynagrodzenie", color="red")
    plt.xlabel("Indeks")
    plt.ylabel("Wynagrodzenie")
    plt.title("Rzeczywiste VS Przewidyane wynagrodzenie")
    plt.legend()
    return plt
