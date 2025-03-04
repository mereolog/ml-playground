import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_data(uploaded_file):
    """Load and preprocess the data from a CSV file."""
    data = pd.read_csv(uploaded_file)
    data = data.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
    x_data = data['YearsExperience'].values.reshape(-1, 1)
    y_data = data['Salary'].values
    return data, x_data, y_data

def train_model(x_data, y_data, model_type, alpha, _epochs=None):
    """Train a Lasso or Ridge regression model."""
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    if model_type == "Lasso":
        model = Lasso(alpha=alpha)
    elif model_type == "Ridge":
        model = Ridge(alpha=alpha)
    else:
        return None, None, None, None

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    cost_function = st.selectbox("Wybierz funkcję kosztu",
                                 ["Błąd średniokwadratowy (MSE)", "Błąd średniobezwzględny (MAE)", "R2 Score"])
    if cost_function == "Błąd średniokwadratowy (MSE)":
        cost = mean_squared_error(y_test, y_pred)
    elif cost_function == "Błąd średniobezwzględny (MAE)":
        cost = mean_absolute_error(y_test, y_pred)
    else:
        cost = r2_score(y_test, y_pred)

    return cost, y_test, y_pred, cost_function

def plot_training_steps(x_data, y_data, learning_rate, epochs, cost_function):
    """Visualize the training process and cost evolution."""
    theta = np.zeros(2)
    x_train_bias = np.c_[np.ones(x_data.shape[0]), x_data]
    cost_history = []

    columns = st.columns(3)

    for epoch in range(epochs):
        predictions = x_train_bias.dot(theta)
        errors = predictions - y_data
        gradient = x_train_bias.T.dot(errors) / len(y_data)
        theta -= learning_rate * gradient

        if cost_function == "Błąd średniokwadratowy (MSE)":
            cost = np.mean(errors ** 2)
        elif cost_function == "Błąd średniobezwzględny (MAE)":
            cost = np.mean(np.abs(errors))
        else:
            cost = 1 - (np.sum(errors ** 2) / np.sum((y_data - np.mean(y_data)) ** 2))
        cost_history.append(cost)

        col_idx = epoch % 3
        with columns[col_idx]:
            st.write(f"Epoka {epoch + 1}, błąd: {cost:.4f}")
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ax2.scatter(x_data, y_data, color='#0080ff', alpha=0.7)
            ax2.plot(x_data, x_train_bias.dot(theta), color='#ff4d4d', linewidth=2)
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
