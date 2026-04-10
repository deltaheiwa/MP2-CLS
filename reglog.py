import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self._classes = None

    def fit(self, x, y):
        self._classes = np.unique(y)
        n_samples, n_features = x.shape

        self.weights = np.zeros((n_samples, n_features))
        self.bias = 0

        for _ in range(self.n_iterations):
            z = np.dot(x, self.weights.T) + self.bias
            y_pred = sigmoid(z)

            dw = (1 / n_samples) * np.dot((y_pred - y).T, x)
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db


    def predict(self, x):
        y_pred = self.predict_proba(x)[:, 1]
        return (y_pred >= 0.5).astype(int)

    def predict_proba(self, x):
        z = np.dot(x, self.weights.T) + self.bias
        y_pred = sigmoid(z)

        return np.vstack((1 - y_pred, y_pred)).T