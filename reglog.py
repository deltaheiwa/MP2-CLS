import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import validate_data, check_is_fitted


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))


class LogisticRegression(ClassifierMixin, BaseEstimator):


    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.__sklearn_is_fitted__ = False
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self._weights = None
        self._bias = 0


    def fit(self, x, y):
        x, y = validate_data(self, x, y)
        self.classes_ = np.unique(y)
        n_samples, n_features = x.shape
        self.n_features_in_ = n_features

        self._weights = np.zeros(n_features)
        self._bias = 0

        for _ in range(self.n_iterations):
            z = np.dot(x, self._weights) + self._bias
            y_pred = sigmoid(z)

            dw = (1 / n_samples) * np.dot(x.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self._weights -= self.learning_rate * dw
            self._bias -= self.learning_rate * db

        self.is_fitted_ = True
        return self


    def predict(self, x):
        y_pred = self.predict_proba(x)[:, 1]
        return (y_pred >= 0.5).astype(int)

    def predict_proba(self, x):
        check_is_fitted(self, 'is_fitted_')

        x = validate_data(self, x, reset=False)
        z = np.dot(x, self._weights.T) + self._bias
        y_pred = sigmoid(z)

        return np.vstack((1 - y_pred, y_pred)).T