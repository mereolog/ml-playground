import numpy as np
import pandas as pd


def load_dataset(file_path):
    try:
        raw_data = pd.read_csv(file_path)
        years_experience = raw_data[raw_data.columns[1]].values
        salary = raw_data[raw_data.columns[2]].values

        return years_experience, salary

    except Exception as e:
        raise Exception(f"Error loading dataset: {e}")


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


def gradient_descent(X, y, learning_rate=0.01, num_iterations=1000,
                     error_func=mean_squared_error, regularization=None, regularization_coefficient=0.1):
    slope, intercept = 0, 0
    n = len(X)

    # Lists to store cost and predictions over time for visualization
    cost_history = []
    predictions_history = []
    r_squared_history = []

    for i in range(num_iterations):
        y_predicted = slope * X + intercept

        if error_func == mean_squared_error:
            m_gradient = (-2 / n) * np.sum(X * (y - y_predicted))
            b_gradient = (-2 / n) * np.sum(y - y_predicted)
        elif error_func == mean_absolute_error:
            m_gradient = (-1 / n) * np.sum(X * np.sign(y - y_predicted))
            b_gradient = (-1 / n) * np.sum(np.sign(y - y_predicted))
        else:
            raise ValueError("Unsupported error function. Use 'm_s_e' or 'm_a_e'.")

        if regularization == 'L1':
            m_gradient += regularization_coefficient * np.sign(slope)
        elif regularization == 'L2':
            m_gradient += regularization_coefficient * slope

        # Update parameters
        slope -= learning_rate * m_gradient
        intercept -= learning_rate * b_gradient

        # Calculate and store cost for this iteration
        cost = error_func(y, y_predicted)
        cost_history.append(cost)
        predictions_history.append((slope, intercept))
        r_s = r_squared(y, y_predicted)
        r_squared_history.append(r_s)

    return slope, intercept, predictions_history, cost_history, r_squared_history


# if __name__ == "__main__":
#
#     # data = pd.read_csv("Salary_dataset.csv")
#     # x = data['YearsExperience'].values
#     # y = data['Salary'].values
#     #
#     # aa, bb, cc, dd, r_hist = gradient_descent(x, y, learning_rate=0.01, num_iterations=1000,
#     #                                           error_func=m_s_e, lambda_reg=0.1)
#     #
#     # print(r_hist[1:5])
