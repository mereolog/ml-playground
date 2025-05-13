import unittest
import numpy as np
from algorithms.supervised.knn import KNearestNeighbor
from configs.k_nearest_neighbour_algorithm import KNNConfig

class TestKNN(unittest.TestCase):
    def test_knn_prediction(self):
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y_train = np.array([0, 0, 1, 1])
        X_test = np.array([[1.5, 2.5], [4.5, 5.5]])

        config = KNNConfig(k=2)
        knn = KNearestNeighbor(config)
        knn.fit(X_train, y_train)

        predictions = knn.predict(X_test)
        expected = np.array([0, 1])

        np.testing.assert_array_equal(predictions, expected)

if __name__ == '__main__':
    unittest.main()