from typing import Callable, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd


def load_dataset(file_path):

    raw_data = pd.read_csv(file_path)
    years_experience = raw_data[raw_data.columns[1]].values
    salary = raw_data[raw_data.columns[2]].values

    return years_experience, salary


def standarization(exp_values, salary_values):

    exp_mean = np.mean(exp_values)
    exp_std = np.std(exp_values)
    salary_mean = np.mean(salary_values)
    salary_std = np.std(salary_values)

    norm_exp = (exp_values - exp_mean) / exp_std
    norm_salary = (salary_values - salary_mean) / salary_std

    return norm_exp, norm_salary


def mean_squared_error(y_actual, y_predicted):
    return np.mean((y_actual - y_predicted) ** 2)


def mean_absolute_error(y_actual, y_predicted):
    return np.mean(np.abs(y_actual - y_predicted))


def r_squared(y_actual, y_predicted):
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - (ss_res / ss_tot)

@dataclass
class Config:
    learning_rate: float = 0.01
    num_iterations: int = 1000
    error_func: Callable = mean_squared_error
    regularization: Optional[str] = None
    regularization_coefficient: float = 0.1


def gradient_descent(x, y, config: Config):
    slope, intercept = 0, 0
    n = len(x)

    # Lists to store cost and predictions over time for visualization
    cost_history = []
    predictions_history = []
    r_squared_history = []

    for _ in range(config.num_iterations):
        y_predicted = slope * x + intercept

        if config.error_func is mean_squared_error:
            m_gradient = (-2 / n) * np.sum(x * (y - y_predicted))
            b_gradient = (-2 / n) * np.sum(y - y_predicted)
        elif config.error_func is mean_absolute_error:
            m_gradient = (-1 / n) * np.sum(x * np.sign(y - y_predicted))
            b_gradient = (-1 / n) * np.sum(np.sign(y - y_predicted))
        else:
            raise ValueError("Use 'mean_squared_error' or 'mean_absolute_error'.")

        if config.regularization == 'L1':
            m_gradient += config.regularization_coefficient * np.sign(slope)
        elif config.regularization == 'L2':
            m_gradient += config.regularization_coefficient * slope

        # Update parameters
        slope -= config.learning_rate * m_gradient
        intercept -= config.learning_rate * b_gradient

        # Calculate and store cost for this iteration
        cost = config.error_func(y, y_predicted)
        cost_history.append(cost)
        predictions_history.append((slope, intercept))
        r_s = r_squared(y, y_predicted)
        r_squared_history.append(r_s)

    return slope, intercept, predictions_history, cost_history, r_squared_history
