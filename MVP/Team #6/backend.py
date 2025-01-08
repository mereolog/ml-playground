import numpy as np
import pandas as pd


def load_dataset(file_path):
    try:
        raw_data = pd.read_csv(file_path)
        x = raw_data[raw_data.columns[1]].values
        y = raw_data[raw_data.columns[2]].values

        return x, y

    except Exception as e:
        raise Exception(f"Error loading dataset: {e}")


def standarization(bad_x, bad_y):
    # Normalizacja normalnie
    exp_mean = np.mean(bad_x)
    exp_std = np.std(bad_x)
    salary_mean = np.mean(bad_y)
    salary_std = np.std(bad_y)

    x = (bad_x - exp_mean) / exp_std
    y = (bad_y - salary_mean) / salary_std

    return x, y


def m_s_e(y_actual, y_predicted):
    return np.mean((y_actual - y_predicted) ** 2)


def m_a_e(y_actual, y_predicted):
    return np.mean(np.abs(y_actual - y_predicted))


def r_squared(y_actual, y_predicted):
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - (ss_res / ss_tot)


def gradient_descent(x, y, learning_rate=0.01, num_iterations=1000,
                     error_func=m_s_e, regularization=None, lambda_reg=0.1):
    m, b = 0, 0
    n = len(x)

    # Lists to store cost and predictions over time for visualization
    cost_history = []
    predictions_history = []
    r_squared_history = []

    for i in range(num_iterations):
        y_predicted = m * x + b

        if error_func == m_s_e:
            m_gradient = (-2 / n) * np.sum(x * (y - y_predicted))
            b_gradient = (-2 / n) * np.sum(y - y_predicted)
        elif error_func == m_a_e:
            m_gradient = (-1 / n) * np.sum(x * np.sign(y - y_predicted))
            b_gradient = (-1 / n) * np.sum(np.sign(y - y_predicted))
        else:
            raise ValueError("Unsupported error function. Use 'm_s_e' or 'm_a_e'.")

        if regularization == 'L1':
            m_gradient += lambda_reg * np.sign(m)
        elif regularization == 'L2':
            m_gradient += lambda_reg * m

        # Update parameters
        m -= learning_rate * m_gradient
        b -= learning_rate * b_gradient

        # Calculate and store cost for this iteration
        cost = error_func(y, y_predicted)
        cost_history.append(cost)
        predictions_history.append((m, b))
        r_s = r_squared(y, y_predicted)
        r_squared_history.append(r_s)

    return m, b, predictions_history, cost_history, r_squared_history


if __name__ == "__main__":
    data = pd.read_csv("Salary_dataset.csv")
    x = data['YearsExperience'].values
    y = data['Salary'].values

    aa, bb, cc, dd, r_hist = gradient_descent(x, y, learning_rate=0.01, num_iterations=1000,
                                              error_func=m_s_e, lambda_reg=0.1)

    print(r_hist[1:5])
