from io import BytesIO
import base64
import os

import matplotlib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, session
import matplotlib.pyplot as plt

matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = os.urandom(24)


def generate_plot(x, y, weights, iteration, column_names):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Dane treningowe')
    pred_y = weights[0] + weights[1] * x
    plt.plot(x, pred_y, color='red', label=f'Linia regresji, krok {iteration}')
    plt.xlabel(column_names[1])
    plt.ylabel(column_names[2])
    plt.legend()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url


def generate_cost_plot(cost_history):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cost_history) + 1), cost_history, marker='o', color='blue')
    plt.title("Zmiana funkcji kosztu w czasie")
    plt.xlabel("Iteracja")
    plt.ylabel("Koszt")
    plt.grid()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url


def perform_single_step(x, y, weights, lr, regularization, reg_param):
    m = len(x)
    y_pred = weights[0] + weights[1] * x
    d_bias = (1 / m) * sum(y_pred - y)
    d_weight = (1 / m) * sum((y_pred - y) * x)

    if regularization == "lasso":
        d_weight += reg_param * np.sign(weights[1])
    elif regularization == "ridge":
        d_weight += reg_param * weights[1]

    weights[0] -= lr * d_bias
    weights[1] -= lr * d_weight
    return weights


def calculate_cost(x, y, weights, cost_function, regularization, reg_param):
    y_pred = weights[0] + weights[1] * x
    cost = None

    if cost_function == "mse":
        cost = np.mean((y - y_pred) ** 2)
    elif cost_function == "mae":
        cost = np.mean(np.abs(y - y_pred))
    elif cost_function == "r2":
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        cost = 1 - (ss_residual / ss_total)

    if regularization == "lasso":
        cost += reg_param * abs(weights[1])
    elif regularization == "ridge":
        cost += reg_param * (weights[1] ** 2)

    return cost


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if 'file' not in request.files:
                return "Nie przesłano pliku"
            file = request.files['file']
            if file.filename == '' or not file.filename.endswith('.csv'):
                return "Prześlij plik w formacie CSV"
            df = pd.read_csv(file)
            if df.shape[1] < 2:
                return "Plik musi zawierać co najmniej dwie kolumny z danymi"

            x = df.iloc[:, 1].values
            y = df.iloc[:, 2].values
            column_names = df.columns.tolist()
            lr = float(request.form["lr"])
            iterations = int(request.form["iterations"])
            cost_function = request.form["cost_function"]
            regularization = request.form["regularization"]
            reg_param = float(request.form["reg_param"])

            session['x'] = x.tolist()
            session['y'] = y.tolist()
            session['lr'] = lr
            session['iterations'] = iterations
            session['current_step'] = 0
            session['column_names'] = column_names
            session['weights'] = [0.0, 0.0]
            session['cost_function'] = cost_function
            session['regularization'] = regularization
            session['reg_param'] = reg_param
            session['cost_history'] = []

            weights = session['weights']
            plot_url = generate_plot(np.array(x), np.array(y), weights, 0, column_names)

            return render_template("index.html", plot=plot_url, step=session['current_step'])

        except Exception as e:
            return str(e)

    return render_template("index.html")


@app.route("/next_step")
def next_step():
    try:
        x = np.array(session['x'])
        y = np.array(session['y'])
        lr = session['lr']
        weights = session['weights']
        cost_function = session['cost_function']
        regularization = session['regularization']
        reg_param = session['reg_param']
        column_names = session.get('column_names', ['x', 'y'])
        current_step = session['current_step']
        iterations = session['iterations']
        cost_history = session['cost_history']

        if current_step >= iterations:

            cost_plot_url = generate_cost_plot(cost_history)
            return render_template("index.html", cost_plot=cost_plot_url, step=current_step, is_finished=True)

        weights = perform_single_step(x, y, weights, lr, regularization, reg_param)
        session['weights'] = weights
        session['current_step'] += 1

        cost = calculate_cost(x, y, weights, cost_function, regularization, reg_param)
        cost_history.append(cost)
        session['cost_history'] = cost_history

        plot_url1 = generate_plot(x, y, weights, session['current_step'], column_names)

        return render_template(
            "index.html",
            plot=plot_url1,
            step=session['current_step'],
            cost=cost,
            cost_function=cost_function,
            is_last_step=(session['current_step'] >= iterations)
        )
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)
