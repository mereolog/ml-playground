import os
import matplotlib.pyplot as plt

def plot_regression(X, y, model):
    # Sprawdzenie, czy folder 'static' istnieje, jeśli nie, to go tworzymy
    if not os.path.exists('static'):
        os.makedirs('static')

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Dane')
    plt.plot(X, model.predict(X), color='red', label='Linia regresji')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.title('Regresja Liniowa')
    plt.legend()
    plt.grid()

    # Zapis wykresu do pliku w folderze 'static'
    plot_path = 'static/regression_plot.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path
