import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st

def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    data = data.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
    X = data['YearsExperience'].values.reshape(-1, 1)
    y = data['Salary'].values
    return data, X, y

def train_model(X, y, model_type, alpha, epochs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "Lasso":
        model = Lasso(alpha=alpha)
    elif model_type == "Ridge":
        model = Ridge(alpha=alpha)
    else:
        return None, None, None, None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cost_function = st.selectbox("Wybierz funkcję kosztu",
                                 ["Błąd średniokwadratowy (MSE)", "Błąd średniobezwzględny (MAE)", "R2 Score"])
    if cost_function == "Błąd średniokwadratowy (MSE)":
        cost = mean_squared_error(y_test, y_pred)
    elif cost_function == "Błąd średniobezwzględny (MAE)":
        cost = mean_absolute_error(y_test, y_pred)
    else:
        cost = r2_score(y_test, y_pred)

    return cost, y_test, y_pred, cost_function

def plot_training_steps(X, y, learning_rate, epochs, cost_function):
    theta = np.zeros(2)
    X_train_bias = np.c_[np.ones(X.shape[0]), X]
    cost_history = []

    columns = st.columns(3)

    for epoch in range(epochs):
        predictions = X_train_bias.dot(theta)
        errors = predictions - y
        gradient = X_train_bias.T.dot(errors) / len(y)
        theta -= learning_rate * gradient

        if cost_function == "Błąd średniokwadratowy (MSE)":
            cost = np.mean(errors ** 2)
        elif cost_function == "Błąd średniobezwzględny (MAE)":
            cost = np.mean(np.abs(errors))
        else:
            cost = 1 - (np.sum(errors ** 2) / np.sum((y - np.mean(y)) ** 2))
        cost_history.append(cost)

        col_idx = epoch % 3
        with columns[col_idx]:
            st.write(f"Epoka {epoch + 1}, błąd: {cost:.4f}")
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ax2.scatter(X, y, color='#0080ff', alpha=0.7)
            ax2.plot(X, X_train_bias.dot(theta), color='#ff4d4d', linewidth=2)
            ax2.set_xlabel("Lata doświadczenia")
            ax2.set_ylabel("Wynagrodzenie")
            ax2.set_title(f"Epoka {epoch + 1} - Predykcje")
            st.pyplot(fig2)

    st.markdown("---")
    st.write("Zmiana błędu w trakcie uczenia:")
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    ax3.plot(range(epochs), cost_history, color='#33cc33')
    ax3.set_xlabel('Epoki')
    ax3.set_ylabel('Błąd')
    ax3.set_title('Zmiana błędu w trakcie uczenia')
    st.pyplot(fig3)