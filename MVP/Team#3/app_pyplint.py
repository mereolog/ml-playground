"""
Flask application for performing linear regression with gradient descent and regularization.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from flask import Flask, render_template, request

matplotlib.use('Agg')

app = Flask(__name__)

def calculate_error_metric(x_values, y_values, slope, intercept, cost_function):
    """Calculates MSE, MAE, or R^2 based on the chosen cost function."""
    y_pred = slope * x_values + intercept
    
    if cost_function == 'MSE':
        return np.mean((y_values - y_pred) ** 2)
    if cost_function == 'MAE':
        return np.mean(np.abs(y_values - y_pred))
    if cost_function == 'R2':
        ss_total = np.sum((y_values - np.mean(y_values)) ** 2)
        ss_res = np.sum((y_values - y_pred) ** 2)
        return 1 - (ss_res / ss_total)
    
    return None

def gradient_descent_with_regularization(x_values, y_values, config):
    """
    Implements gradient descent with Lasso or Ridge regularization.
    """
    slope, intercept = 0, 0
    num_samples = len(y_values)
    learning_rate = config['learning_rate']
    iterations = config['iterations']
    regularization_type = config['regularization_type']
    regularization_param = config['regularization_param']
    cost_function = config['cost_function']
    cost_history = []
    
    for _ in range(iterations):
        y_pred = slope * x_values + intercept
        gradient_slope = (-2 / num_samples) * np.sum(x_values * (y_values - y_pred))
        gradient_intercept = (-2 / num_samples) * np.sum(y_values - y_pred)
        
        if regularization_type == 'Lasso':
            gradient_slope += regularization_param * np.sign(slope)
        elif regularization_type == 'Ridge':
            gradient_slope += regularization_param * slope
        
        slope -= learning_rate * gradient_slope
        intercept -= learning_rate * gradient_intercept
        cost_history.append(calculate_error_metric(x_values, y_values, slope, intercept, cost_function))
    
    return slope, intercept, cost_history

def plot_regression(x_values, y_values, slope, intercept):
    """Creates and saves a regression plot."""
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, color='blue', label='Data')
    plt.plot(x_values, slope * x_values + intercept, color='red', label='Regression Line')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.title('Linear Regression with Gradient Descent')
    plt.legend()
    plt.grid()
    plot_path = 'static/years_salary_linear_regression_plot.png'
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    return plot_path

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles form submission and dataset processing."""
    if request.method == 'POST':
        error_message = None
        
        try:
            learning_rate = float(request.form['learning_rate'])
            iterations = int(request.form['iterations'])
            regularization_type = request.form['regularization_type']
            regularization_param = float(request.form['regularization_param'])
            cost_function = request.form['cost_function']
        except ValueError:
            return render_template('index.html', error_message="Invalid numerical input.")
        
        uploaded_dataset = request.files.get('dataset')
        try:
            data = pd.read_csv(uploaded_dataset) if uploaded_dataset else pd.read_csv('Salary_dataset.csv')
        except (pd.errors.ParserError, FileNotFoundError, Exception) as e:
            return render_template('index.html', error_message=f"Dataset error: {str(e)}")
        
        x_column, y_column = request.form.get('x_column'), request.form.get('y_column')
        if x_column not in data.columns or y_column not in data.columns:
            return render_template('index.html', error_message="Invalid column selection.")
        
        x_values = data[x_column].values.reshape(-1, 1).flatten()
        y_values = data[y_column].values
        
        slope, intercept, cost_history = gradient_descent_with_regularization(
            x_values, y_values, config={
                'learning_rate': learning_rate,
                'iterations': iterations,
                'regularization_type': regularization_type,
                'regularization_param': regularization_param,
                'cost_function': cost_function
            }
        )
        
        plot_path = plot_regression(x_values, y_values, slope, intercept)
        return render_template(
            'index.html',
            plot_path=plot_path,
            slope=slope,
            intercept=intercept,
            cost_history=cost_history,
            cost_function=cost_function,
            x_column=x_column,
            y_column=y_column
        )
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
