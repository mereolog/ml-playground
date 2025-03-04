import numpy as np
import pandas as pd


def load_dataset(file_path):
    try:
        raw_data = pd.read_csv(file_path)
        years_experience = raw_data[raw_data.columns[1]].values
        salary = raw_data[raw_data.columns[2]].values

        return years_experience, salary
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Error loading dataset: {error}")


def standardize_data(exp_values, salary_values):
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
    ss_residual = np.sum((y_actual - y_predicted) ** 2)
    ss_total = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - (ss_residual / ss_total)


def gradient_descent(features, target, learning_rate=0.01, num_iterations=1000,
                     error_func=mean_squared_error, regularization=None, reg_coeff=0.1)
    slope, intercept = 0, 0
    n_samples = len(features)

    cost_history = [] 
    predictions_history = []
    r_squared_history = []

    for iteration in range(num_iterations):
        y_predicted = slope * features + intercept

        if error_func == mean_squared_error:
            m_gradient = (-2 / n_samples) * np.sum(features * (target - y_predicted))
            b_gradient = (-2 / n_samples) * np.sum(target - y_predicted)
        elif error_func == mean_absolute_error:
            m_gradient = (-1 / n_samples) * np.sum(features * np.sign(target - y_predicted))
            b_gradient = (-1 / n_samples) * np.sum(np.sign(target - y_predicted))
        else:
            raise ValueError("Unsupported error function. Use 'mean_squared_error' or 'mean_absolute_error'.")

        if regularization == 'L1':
            m_gradient += reg_coeff * np.sign(slope)
        elif regularization == 'L2':
            m_gradient += reg_coeff * slope

        slope -= learning_rate * m_gradient
        intercept -= learning_rate * b_gradient
        
        cost = error_func(target, y_predicted)
        cost_history.append(cost)
        predictions_history.append((slope, intercept))
        r_s = r_squared(target, y_predicted)
        r_squared_history.append(r_s)

    return slope, intercept, predictions_history, cost_history, r_squared_history
