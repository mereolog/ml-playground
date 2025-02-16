import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from model_logic import load_data, train_model, calculate_cost, plot_training_steps

st.set_page_config(page_title="Regresja liniowa", layout="wide")

st.markdown("<h1 style='text-align: center; color: #006699;'>Regresja liniowa</h1>", unsafe_allow_html=True) #blue color

uploaded_file = st.file_uploader("Prześlij plik CSV", type="csv", key="fileUploader")

if uploaded_file:
    data, X, y = load_data(uploaded_file)
    with st.container():
        st.markdown("<h3 style='color: #003366;'>Podgląd danych:</h3>", unsafe_allow_html=True) #navy blue color
        st.dataframe(pd.DataFrame({
            'Doświadczenie': X.flatten(),
            'Wynagrodzenie': y.astype(str)
        }), height=150)

    st.markdown("---")

    cost_function = st.selectbox("Wybierz funkcję kosztu", ["Błąd średniokwadratowy (MSE)", "Błąd średniobezwzględny (MAE)", "R2 Score"])
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        regularization_type = st.selectbox("Wybierz typ regularyzacji", ["None", "Lasso", "Ridge"])
    with col2:
        alpha = st.slider("Wybierz parametr regularyzacji (alpha)", 0.01, 10.0, 1.0) #min_value, max_value, default_value
    with col3:
        learning_rate = st.slider("Wybierz współczynnik uczenia", 0.001, 0.1, 0.01) #min_value, max_value, default_value

    epochs = st.slider("Wybierz liczbę epok", 1, 10, 1) #min_value, max_value, default_value

    if 'YearsExperience' not in data.columns or 'Salary' not in data.columns:
        st.error("Dane muszą zawierać kolumny 'YearsExperience' i 'Salary'.")
    else:
        y_test, y_pred = train_model(X, y, regularization_type, alpha)

        if y_test is not None and y_pred is not None:
            cost = calculate_cost(y_test, y_pred, cost_function)
            st.success(f"Wartość funkcji kosztu ({cost_function}): {cost:.4f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Dane rzeczywiste', marker=dict(color='#0080ff')))
            fig.update_layout(title='Rzeczywiste vs Przewidywane wyniki', xaxis_title='Rzeczywiste wynagrodzenie', yaxis_title='Przewidywane wynagrodzenie')
            st.plotly_chart(fig)

    plot_training_steps(X, y, learning_rate, epochs, cost_function)
