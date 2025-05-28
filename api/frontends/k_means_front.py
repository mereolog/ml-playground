import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Button

from algorithms.unsupervised.k_means import KMeans


def check_numeric_columns(df, columns):
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Kolumna '{col}' nie jest numeryczna!")


class KMeansVisualizer:
    def __init__(self, kmeans, X):
        self.kmeans = kmeans
        self.X = X
        self.index = 0
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        plt.subplots_adjust(bottom=0.25)

        self.button_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.button = Button(self.button_ax, 'Następny krok')
        self.button.on_clicked(self.next_step)

        self.plot_step()

    def plot_step(self):
        self.ax.clear()
        centroids, labels, inertia = self.kmeans.history[self.index]
        for i in range(self.kmeans.k):
            self.ax.scatter(self.X[labels == i, 0], self.X[labels == i, 1], label=f'Cluster {i}')
        self.ax.scatter(centroids[:, 0], centroids[:, 1], edgecolors='black', c='white', marker='X', s=50, label='Centroids')

        self.ax.set_title(f'Iteracja {self.index}')
        inertia_text = f"Inertia: {inertia:.2f}"
        self.ax.text(0.5, -0.07, inertia_text, transform=self.ax.transAxes,
                     ha='center', va='top', fontsize=10, color='gray')

        self.ax.legend()
        self.ax.grid(True)
        self.fig.canvas.draw()

    def next_step(self, event):
        if self.index < len(self.kmeans.history) - 1:
            self.index += 1
            self.plot_step()


df = pd.read_csv("datasets/Live.csv")

columns_to_use = ['num_reactions', 'num_likes']
check_numeric_columns(df, columns_to_use)

X = df[columns_to_use].values

kmeans = KMeans(k=3, max_iters=10, init_method='k-means++', metric='euclidean')
kmeans.fit(X)

visualizer = KMeansVisualizer(kmeans, X)
plt.show()
