import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
import plotly.graph_objects as go

def load_data(uploaded_file):
    """Loads CSV data and processes it for training."""
    data = pd.read_csv(uploaded_file)
    data = data.applymap(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
    x = data['YearsExperience'].values.reshape(-1, 1)
    y = data['Salary'].values
    return data, x, y

def train_model(x, y, regularization_type, alpha):
    """Trains a regression model with specified regularization."""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    if regularization_type == "Lasso":
        model = Lasso(alpha=alpha)
    elif regularization_type == "Ridge":
        model = Ridge(alpha=alpha)
    else:
        model = LinearRegression()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return y_test, y_pred

def calculate_cost(y_test, y_pred, cost_function):
    """Calculates the cost based on the specified function."""
    if cost_function == "Błąd średniokwadratowy (MSE)":
        cost = mean_squared_error(y_test, y_pred)
    elif cost_function == "Błąd średniobezwzględny (MAE)":
        cost = mean_absolute_error(y_test, y_pred)
    else:
        cost = r2_score(y_test, y_pred)
    return cost

def plot_training_steps(x, y, learning_rate, epochs, cost_function):
    """Plots training steps and cost function over epochs."""
    theta = np.zeros(2)
    x_train_bias = np.c_[np.ones(x.shape[0]), x]
    cost_history = []

    for epoch in range(epochs):
        predictions = x_train_bias.dot(theta)
        errors = predictions - y
        gradient = x_train_bias.T.dot(errors) / len(y)
        theta -= learning_rate * gradient

        if cost_function == "Błąd średniokwadratowy (MSE)":
            cost = np.mean(errors**2)
        elif cost_function == "Błąd średniobezwzględny (MAE)":
            cost = np.mean(np.abs(errors))
        else:
            cost = 1 - (np.sum(errors**2) / np.sum((y - np.mean(y)) ** 2))
        cost_history.append(cost)

        st.write(f"Epoka {epoch + 1}, koszt: {cost:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x.flatten(), y=y, mode='markers', name='Rzeczywiste',
            marker={"color": '#0080ff'}
        ))
        fig.add_trace(go.Scatter(
            x=x.flatten(), y=x_train_bias.dot(theta), mode='lines', name='Predykcje',
            line={"color": '#ff4d4d'}
        ))
        fig.update_layout(
            title=f'Epoka {epoch + 1} - Predykcje',
            xaxis_title='Lata doświadczenia',
            yaxis_title='Wynagrodzenie'
        )
        st.plotly_chart(fig)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(epochs)), y=cost_history, mode='lines', name='Błąd',
        line={"color": '#33cc33'}
    ))
    fig.update_layout(
        title='Zmiana błędu w trakcie uczenia',
        xaxis_title='Epoki',
        yaxis_title='Błąd'
    )
    st.plotly_chart(fig)
