import unittest
import numpy as np
from algorithms.supervised.naive_bayes_classificator import NaiveBernoulliClassifier
from schemas.configs.naive_bayes_configs import NaiveBayesParams


class TestNaiveBernoulliClassifier(unittest.TestCase):

    def setUp(self):
        """Set up test data and initialize the classifier."""
        self.X_train = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]])
        self.y_train = np.array([1, 0, 1, 0])
        self.X_test = np.array([[1, 0, 0], [0, 1, 1]])
        self.y_test = np.array([1, 0])

        self.classifier = NaiveBernoulliClassifier(params=NaiveBayesParams(verbose=True))

    def test_fit(self):
        """Test the fit method."""
        self.classifier.fit(self.X_train, self.y_train)

        self.assertIsNotNone(self.classifier.feature_probs, "Feature probabilities should not be None after fitting.")
        self.assertIsNotNone(self.classifier.class_probs, "Class probabilities should not be None after fitting.")

    def test_predict_proba(self):
        """Test the predict_proba method."""
        self.classifier.fit(self.X_train, self.y_train)
        probs = self.classifier.predict_proba(self.X_test)

        self.assertEqual(probs.shape, (2, 2), "Predict_proba should return an array of shape (n_samples, n_classes).")
        self.assertTrue(np.all(probs >= 0) and np.all(probs <= 1), "Probabilities should be between 0 and 1.")

    def test_predict(self):
        """Test the predict method."""
        self.classifier.fit(self.X_train, self.y_train)
        predictions = self.classifier.predict(self.X_test)

        self.assertEqual(predictions.shape, (2,), "Predict should return an array of shape (n_samples,).")
        self.assertTrue(np.all(np.isin(predictions, [0, 1])), "Predictions should be binary (0 or 1).")

    def test_score(self):
        """Test the score method."""
        self.classifier.fit(self.X_train, self.y_train)
        scores = self.classifier.score(self.X_test, self.y_test)

        self.assertIn("accuracy", scores, "Score dictionary should contain 'accuracy'.")
        self.assertIn("log_loss", scores, "Score dictionary should contain 'log_loss'.")
        self.assertGreaterEqual(scores["accuracy"], 0, "Accuracy should be >= 0.")
        self.assertGreaterEqual(scores["log_loss"], 0, "Log loss should be >= 0.")

    def test_get_parameters(self):
        """Test the get_parameters method."""
        self.classifier.fit(self.X_train, self.y_train)
        params = self.classifier.get_parameters()

        self.assertIn("class_probs", params, "Parameters should contain 'class_probs'.")
        self.assertIn("feature_probs", params, "Parameters should contain 'feature_probs'.")
        self.assertEqual(params["class_probs"].shape, (2,), "Class probabilities should have shape (n_classes,).")
        self.assertEqual(params["feature_probs"].shape, (2, self.X_train.shape[1]),
                         "Feature probabilities should have shape (n_classes, n_features).")


if __name__ == "__main__":
    unittest.main()