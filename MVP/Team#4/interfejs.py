import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import load_data, train_model, plot_training_steps

PAGE_TITLE = "Linear Regression"
HEADER_TEXT = "Linear Regression"
HEADER_COLOR = "#006699" #Dark Blue

st.set_page_config(page_title=PAGE_TITLE, layout="wide")

#Add styled header using HTML
st.markdown(f"<h1 style='text-align: center; color: {HEADER_COLOR};'>{HEADER_TEXT}</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Prześlij plik CSV", type="csv", key="fileUploader")

if uploaded_file:
    data, X, y = load_data(uploaded_file)

    with st.container():
        st.markdown("<h3 style='color: #003366;'>Podgląd danych:</h3>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            'Doświadczenie': X.flatten(),
            'Wynagrodzenie': y.astype(str)
        }), height=150)

    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        model_type = st.selectbox("Wybierz typ modelu", ["None", "Lasso", "Ridge"])
    with col2:
        alpha = st.slider("Wybierz parametr regularizacji (alpha)", 0.01, 10.0, 1.0)
    with col3:
        learning_rate = st.slider("Wybierz współczynnik uczenia", 0.001, 0.1, 0.01)

    epochs = st.slider("Wybierz liczbę epok", 1, 10, 1)

    if 'YearsExperience' not in data.columns or 'Salary' not in data.columns:
        st.error("Dane muszą zawierać kolumny 'YearsExperience' i 'Salary'.")
    else:
        cost, y_test, y_pred, cost_function = train_model(X, y, model_type, alpha, epochs)

        if cost is not None:
            st.success(f"Wartość funkcji kosztu ({cost_function}): {cost:.4f}")

            fig, ax = plt.subplots(figsize=(5, 3))
            ax.scatter(y_test, y_pred, color='#0080ff', label='Dane rzeczywiste', alpha=0.7)
            ax.set_xlabel("Rzeczywiste wynagrodzenie")
            ax.set_ylabel("Przewidywane wynagrodzenie")
            ax.set_title("Rzeczywiste vs Przewidywane wyniki")
            ax.legend()
            st.pyplot(fig)

        if st.button("Uruchom uczenie krok po kroku"):
            plot_training_steps(X, y, learning_rate, epochs, cost_function)