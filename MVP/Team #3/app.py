import io
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge

from flask import Flask, render_template, request, send_file


app = Flask(__name__)

# Function calculates MSE, MAE, R^2
def calculate_error_metric(X, y, m, b, cost_function):
    y_pred = m * X + b
    if cost_function == 'MSE':
        return np.mean((y - y_pred) ** 2)
    elif cost_function == 'MAE':
        return np.mean(np.abs(y - y_pred))
    elif cost_function == 'R2':
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - (ss_res / ss_total)

def gradient_descent_with_regularization(X, y, config):
    """
    Implements gradient descent with Lasso or Ridge regularization.

    Parameters:
        X (numpy.ndarray): Input feature array of shape (n_samples,).
        y (numpy.ndarray): Target values of shape (n_samples,).
        config (dict): Configuration dictionary containing:
            - learning_rate (float): The step size for gradient descent updates.
            - iterations (int): Number of iterations to run gradient descent.
            - regularization_type (str): Type of regularization ('Lasso' or 'Ridge').
            - regularization_param (float): Regularization parameter (lambda).
            - cost_function (callable): Function to compute the cost/loss.

    Returns:
        tuple: (m, b, cost_history)
            - m (float): Slope of the regression line.
            - b (float): Intercept of the regression line.
            - cost_history (list): List of cost values for each iteration.
    """
    # Initialize slope (m) and intercept (b) to zero
    m = 0  # Slope of the regression line
    b = 0  # Intercept of the regression line
    n = len(y)  # Number of data points

    # Extract configuration parameters
    learning_rate = config['learning_rate']
    iterations = config['iterations']
    regularization_type = config['regularization_type']
    regularization_param = config['regularization_param']
    cost_function = config['cost_function']

    # Store cost history for analysis
    cost_history = []

    for _ in range(iterations):
        # Predicted values
        y_pred = m * X + b

        # Gradients for m and b
        dm = (-2 / n) * np.sum(X * (y - y_pred))  # Gradient w.r.t. slope
        db = (-2 / n) * np.sum(y - y_pred)  # Gradient w.r.t. intercept

        # Apply regularization to the gradient of m
        if regularization_type == 'Lasso':
            dm += regularization_param * np.sign(m)  # L1 penalty
        elif regularization_type == 'Ridge':
            dm += regularization_param * m  # L2 penalty

        # Update parameters using gradients
        m -= learning_rate * dm
        b -= learning_rate * db

        # Compute and store the cost
        cost = cost_function(X, y, m, b)
        cost_history.append(cost)

    return m, b, cost_history


# Create a regression plot
def plot_regression(X, y, m, b):

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Dane')
    plt.plot(X, m * X + b, color='red', label='Linia regresji')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.title('Regresja Liniowa z Gradient Descent')
    plt.legend()
    plt.grid()

    # Ensure the directory exists before saving
    plot_path = 'static/years_salary_linear_regression_plot.png'
    try:
        # Check and create the directory if it doesn't exist
        directory = os.path.dirname(plot_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the plot
        plt.savefig(plot_path)
    except FileNotFoundError:
        return "Error: The file path for saving the plot is invalid."
    except PermissionError:
        return "Error: Insufficient permissions to save the plot."
    except Exception as e:
        return f"An unexpected error occurred while saving the plot: {str(e)}"
    finally:
        # Always close the plot to free up memory
        plt.close()

    return plot_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Parse and cast form inputs
            learning_rate = float(request.form['learning_rate'])
            iterations = int(request.form['iterations'])
            regularization_type = request.form['regularization_type']
            regularization_param = float(request.form['regularization_param'])

            # Proceed with the main logic
            uploaded_dataset = request.files.get('dataset')

            try:
                # Load dataset
                data = pd.read_csv(uploaded_dataset) if uploaded_dataset else pd.read_csv('Salary_dataset.csv')
            except pd.errors.ParserError:
                return "Error: Uploaded file is not a valid CSV format."
            except FileNotFoundError:
                return "Error: Default dataset 'Salary_dataset.csv' not found."
            except Exception as e:
                return f"An error occurred while loading the dataset: {str(e)}"

            # Ensure required columns are present
            if 'YearsExperience' not in data.columns or 'Salary' not in data.columns:
                return "Error: Dataset must contain 'YearsExperience' and 'Salary' columns."

            # Extract features and target
            X = data['YearsExperience'].values.reshape(-1, 1)
            y = data['Salary'].values

            # Perform gradient descent
            m, b, cost_history = gradient_descent_with_regularization(
                X.flatten(),
                y,
                config={
                    'learning_rate': learning_rate,
                    'iterations': iterations,
                    'regularization_type': regularization_type,
                    'regularization_param': regularization_param,
                    'cost_function': calculate_error_metric
                }
            )

            # Generate regression plot
            plot_path = plot_regression(X, y, m, b)

            return render_template(
                'index.html',
                plot_path=plot_path,
                m=m,
                b=b,
                cost_history=cost_history
            )
        except ValueError as e:
            return f"Error: Invalid input - {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

    return render_template('index.html', plot_path=None, cost_history=None)


if __name__ == '__main__':
    app.run(debug=True)
