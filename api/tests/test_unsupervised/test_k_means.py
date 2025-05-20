import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from api.algorithms.unsupervised.k_means import KMeans
import numpy as np
import pandas as pd


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


df = pd.read_csv("C:\ml_pl\pythonProject1\ml-playground\datasets\Live.csv")
X = df[['num_reactions', 'num_likes']].values

kmeans = KMeans(k=3, max_iters=10, init_method='k-means++', metric='euclidean')
kmeans.fit(X)

visualizer = KMeansVisualizer(kmeans, X)
plt.show()