import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, session, redirect, url_for
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
app.secret_key = os.urandom(24)


def generate_plot(x, y, weights, iteration):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Dane treningowe')
    pred_y = weights[0] + weights[1] * x
    plt.plot(x, pred_y, color='red', label=f'Linia regresji, krok {iteration}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url


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
            lr = float(request.form["lr"])
            iterations = int(request.form["iterations"])

            session['x'] = x.tolist()
            session['y'] = y.tolist()
            session['lr'] = lr
            session['iterations'] = iterations
            session['current_step'] = 0

            model = LinearRegression()
            x_reshaped = np.array(x).reshape(-1, 1)
            model.fit(x_reshaped, y)

            session['weights'] = [model.intercept_, model.coef_[0].tolist()]

            plot_url = generate_plot(x, y, session['weights'], 1)
            session['current_step'] += 1

            return render_template("index.html", plot=plot_url, step=session['current_step'])
        except Exception as e:
            return str(e)

    return render_template("index.html")


@app.route("/next_step")
def next_step():
    try:
        x = np.array(session['x']).reshape(-1, 1)
        y = np.array(session['y'])
        current_step = session['current_step']
        if current_step >= session['iterations']:
            return "Uczenie zakończone"

        model = LinearRegression()
        model.intercept_ = session['weights'][0]
        model.coef_ = np.array([session['weights'][1]])

        model.fit(x, y)

        session['weights'] = [model.intercept_, model.coef_[0].tolist()]

        session['current_step'] += 1

        plot_url = generate_plot(x.flatten(), y, session['weights'], session['current_step'])
        return render_template("index.html", plot=plot_url, step=session['current_step'])
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)
