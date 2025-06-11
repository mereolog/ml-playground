# streamlit run interface/app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from mpl_toolkits.mplot3d import Axes3D  # For 3D plot

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.supervised.logistic_regression import LogisticRegression
from algorithms.supervised.polynomial_regression import PolynomialRegression

from schemas.configs.logistic_regression import LogisticRegressionParams
from schemas.configs.polynomial_regression import PolynomialRegressionParams

from utils.metrics import (
    accuracy_score,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

st.set_page_config(page_title="ML Playground", layout="wide")
st.title("Logistic & Polynomial Regression")

model_type = st.sidebar.selectbox("Select model", ["Logistic Regression", "Polynomial Regression"])

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    all_columns = list(data.columns)
    target_column = st.selectbox("Select target variable", all_columns)
    feature_columns = st.multiselect("Select feature columns", [col for col in all_columns if col != target_column])

    if feature_columns and target_column:
        data_encoded = data.copy()
        for col in feature_columns:
            if data[col].dtype == object:
                data_encoded[col] = LabelEncoder().fit_transform(data[col])

        X = data_encoded[feature_columns].values
        y = data[target_column].values

        if model_type == "Logistic Regression":
            if y.dtype.kind in {"O", "U", "S"}:
                unique_labels = pd.Series(y).unique()
                if len(unique_labels) != 2:
                    st.error("Logistic regression supports only binary classification.")
                    st.stop()
                label_map = {label: i for i, label in enumerate(unique_labels)}
                y = np.array([label_map[val] for val in y])

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        if model_type == "Logistic Regression":
            st.sidebar.markdown("Logistic Regression Parameters")
            lr = st.sidebar.number_input("Learning rate", 0.0001, 1.0, 0.01)
            epochs = st.sidebar.number_input("Epochs", 1, 1000, 100)
            batch_size = st.sidebar.number_input("Batch size (0 = full batch)", 0, 512, 0)
            regularization = st.sidebar.selectbox("Regularization", [None, "l1", "l2"])
            lambda_ = st.sidebar.number_input("Lambda", 0.0, 10.0, 0.01)
            threshold = st.sidebar.slider("Decision threshold", 0.01, 0.99, 0.5)

            params = LogisticRegressionParams(
                learning_rate=lr,
                epochs=epochs,
                batch_size=None if batch_size == 0 else batch_size,
                regularization=regularization,
                lambda_=lambda_,
                threshold=threshold,
            )
            model = LogisticRegression(params)

        else: 
            st.sidebar.markdown("Polynomial Regression Parameters")
            lr = st.sidebar.number_input("Learning rate", 0.0001, 1.0, 0.01)
            epochs = st.sidebar.number_input("Epochs", 1, 1000, 100)
            batch_size = st.sidebar.number_input("Batch size (0 = full batch)", 0, 512, 0)
            degree = st.sidebar.slider("Polynomial degree", 1, 10, 2)
            include_bias = st.sidebar.checkbox("Include bias", True)
            loss = st.sidebar.selectbox("Loss function", ["mse", "mae"])
            reg_type = st.sidebar.selectbox("Regularization", [None, "l1", "l2"])
            reg_strength = st.sidebar.number_input("Regularization strength", 0.0, 10.0, 0.01)
            mixing_ratio = st.sidebar.slider("Mixing ratio (if applicable)", 0.0, 1.0, 0.5)

            params = PolynomialRegressionParams(
                learning_rate=lr,
                epochs=epochs,
                batch_size=None if batch_size == 0 else batch_size,
                degree=degree,
                include_bias=include_bias,
                loss=loss,
                reg_type=reg_type,
                reg_strength=reg_strength,
                mixing_ratio=mixing_ratio,
            )
            model = PolynomialRegression(params)

        if st.button("Train Model"):
            model.fit(X, y)
            st.success("Model successfully trained!")

            if model_type == "Logistic Regression":
                y_pred = model.predict(X)
                y_pred_prob = model._predict_raw(X)
                scores = model.score(X, y)

                st.subheader("Classification Metrics")
                st.metric("Accuracy", round(scores["accuracy"], 4))
                st.metric("Log Loss", round(scores["log_loss"], 4))

                st.subheader("Model Coefficients")
                st.json(model.get_coefficients())

                st.subheader("Loss over Epochs")
                st.line_chart(model.get_training_history()["training_loss"])

            else:
                y_pred = model.predict(X)
                st.subheader("Regression Metrics")
                st.metric("MSE", round(mean_squared_error(y, y_pred), 4))
                st.metric("MAE", round(mean_absolute_error(y, y_pred), 4))
                st.metric("R²", round(r2_score(y, y_pred), 4))

                st.subheader("Model Coefficients")
                st.json(model.get_coefficients())

                st.subheader("Loss over Epochs")
                st.line_chart(model.get_training_history()["training_loss"])

                # 3D plot for 2 features
                if X.shape[1] == 2:
                    st.subheader("3D Prediction Surface")
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection="3d")

                    x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
                    y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
                    xx, yy = np.meshgrid(x_range, y_range)
                    grid = np.c_[xx.ravel(), yy.ravel()]
                    zz = model.predict(grid).reshape(xx.shape)

                    ax.plot_surface(xx, yy, zz, cmap="viridis", alpha=0.6)
                    ax.scatter(X[:, 0], X[:, 1], y, color="red")

                    ax.set_xlabel(feature_columns[0])
                    ax.set_ylabel(feature_columns[1])
                    ax.set_zlabel("Target")

                    st.pyplot(fig)


            if X.shape[1] == 1:
                st.subheader("Prediction Plot")
                fig, ax = plt.subplots()
                ax.scatter(X, y, label="True values")
                ax.plot(X, y_pred, color="red", label="Predictions")
                ax.legend()
                st.pyplot(fig)
