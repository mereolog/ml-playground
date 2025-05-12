import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from api.algorithms.unsupervised.k_means import KMeans
import numpy as np


class KMeansVisualizer:
    def __init__(self, kmeans, X):
        self.kmeans = kmeans
        self.X = X
        self.index = 0
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        plt.subplots_adjust(bottom=0.2)

        self.button_ax = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.button = Button(self.button_ax, 'Następny krok')
        self.button.on_clicked(self.next_step)

        self.plot_step()

    def plot_step(self):
        self.ax.clear()
        centroids, labels = self.kmeans.history[self.index]
        for i in range(self.kmeans.k):
            self.ax.scatter(self.X[labels == i, 0], self.X[labels == i, 1], label=f'Cluster {i}')
        self.ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroids')
        self.ax.set_title(f'Iteracja {self.index}')
        self.ax.legend()
        self.ax.grid(True)
        self.fig.canvas.draw()

    def next_step(self, event):
        if self.index < len(self.kmeans.history) - 1:
            self.index += 1
            self.plot_step()


np.random.seed(42)

X = np.vstack((
    np.random.normal([2, 2], 0.4, size=(50, 2)),
    np.random.normal([7, 7], 0.4, size=(50, 2)),
    np.random.normal([2, 7], 0.4, size=(50, 2))
))

kmeans = KMeans(k=4, init_method='random', max_iters=6, tol=0)
kmeans.fit(X)

visualizer = KMeansVisualizer(kmeans, X)
plt.show()