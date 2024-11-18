import streamlit as st
import pandas as pd
from model_logic import load_data, train_model, plot_errors, plot_predictions

st.title("Regresja liniowa do przewidywania wynagrodzeń")
st.write("Ta aplikacja przewiduje wynagrodzenie na podstawie doświadczenia zawodowego za pomocą regresji liniowej.")

if 'epoch' not in st.session_state:
    st.session_state.epoch = 0
if 'errors' not in st.session_state:
    st.session_state.errors = []

uploaded_file = st.file_uploader("Prześlij plik CSV", type="csv")
if uploaded_file:
    X, y = load_data(uploaded_file)

    st.subheader("Podgląd danych")
    st.write(pd.DataFrame({
        'Doświadczenie': X.values.flatten(),
        'Wynagrodzenie': y.astype(str)  
    }))


    st.subheader("Parametry modelu")
    cost_function = st.selectbox("Wybierz funkcję kosztu", ["MSE", "MAE", "R2"])
    reg_type = st.selectbox("Wybierz rodzaj regularyzacji", [None, "Ridge", "Lasso"])
    reg_param = st.slider("Parametr regularyzacji (alpha)", 0.0, 1.0, 0.1) if reg_type else 0
    learning_rate = st.slider("Wskaźnik uczenia (Learning Rate)", 0.001, 1.0, 0.01)
    epochs = st.slider("Liczba epok", 1, 10, 1)
    step_by_step = st.checkbox("Wykonaj krok po kroku")

    if st.button("Trenuj model"):
        with st.spinner("Trwa trenowanie modelu..."):
            if step_by_step:
                step_generator = train_model(X, y, reg_type, reg_param, epochs, cost_function, step_by_step=True, learning_rate=learning_rate)
                for epoch, error, y_test, y_pred in step_generator:
                    if epoch == st.session_state.epoch:
                        st.session_state.errors.append(error)
                        st.write(f"Epoka {epoch + 1}: {cost_function} = {error}")
                        st.session_state.epoch += 1
                        break
                st.button("Kontynuuj")
            else:
                model, errors, y_test, y_pred = train_model(X, y, reg_type, reg_param, epochs, cost_function, step_by_step=False, learning_rate=learning_rate)
                st.session_state.errors = errors


            st.subheader("Zmiana błędu epoka po epoce")
            st.pyplot(plot_errors(st.session_state.errors))

            st.subheader("Rzeczywiste VS Przewidywane wynagrodzenie")
            st.pyplot(plot_predictions(y_test, y_pred))
else:
    st.info("Prześlij plik CSV, aby rozpocząć.")