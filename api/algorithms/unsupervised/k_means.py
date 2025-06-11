import numpy as np


class KMeans:
    def __init__(self, k=3, max_iters=100, tol=0,
                 init_method='random', metric='euclidean'):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.init_method = init_method
        self.metric = metric

        self.inertia_ = None
        self.centroids = None
        self.labels = None
        self.history = []

    def _initialize_centroids(self, X):
        if self.init_method == 'random':
            indices = np.random.choice(len(X), self.k, replace=False)
            return X[indices]
        elif self.init_method == 'k-means++':
            centroids = [X[np.random.randint(len(X))]]
            for _ in range(1, self.k):
                distances = np.min(self._compute_distances(X, np.array(centroids)), axis=1)
                probs = distances ** 2
                probs /= np.sum(probs)
                next_index = np.random.choice(len(X), p=probs)
                centroids.append(X[next_index])
            return np.array(centroids)
        else:
            raise ValueError("Invalid init_method. Use 'random' or 'k-means++'.")

    def _compute_distances(self, X, centroids):
        if self.metric == 'euclidean':
            return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        elif self.metric == 'manhattan':
            return np.sum(np.abs(X[:, np.newaxis] - centroids), axis=2)
        else:
            raise ValueError("Invalid metric. Use 'euclidean' or 'manhattan'.")

    def fit(self, X, visualize=False):
        self.centroids = self._initialize_centroids(X)

        for iteration in range(self.max_iters):
            distances = self._compute_distances(X, self.centroids)
            labels = np.argmin(distances, axis=1)

            step_inertia = np.sum(np.min(distances, axis=1) ** 2)
            self.history.append((self.centroids.copy(), labels.copy(), step_inertia))

            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i) else self.centroids[i]
                for i in range(self.k)
            ])

            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids
            self.labels = labels

            if shift < self.tol:
                break

        self.inertia_ = step_inertia

    def predict(self, X):
        distances = self._compute_distances(X, self.centroids)
        return np.argmin(distances, axis=1)
