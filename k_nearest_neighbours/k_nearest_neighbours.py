import numpy as np
from sklearn.preprocessing import scale


class KNN:
    def __init__(self, k, n_classes):
        self.k = k
        self.n_classes = n_classes

    def prob(self, x):
        """Probability vector for x belonging to any class"""
        # Compute indexes of the k nearest neighbours
        distances = [np.linalg.norm(x_train - x) for x_train in self.X_train]
        k_neighbour_idxs = np.argsort(distances)[:self.k]
        ps = [sum(self.y_train[k_neighbour_idxs] == c) / self.k for c in range(self.n_classes)]

        return ps

    def fit(self, X, y):
        # Just store the training data since KNN is non-parametric
        # It is important to scale the data otherwise the euclidean distance measure favours
        # dimensions with higher absolute values
        self.X_train = scale(X)
        self.y_train = y

    def predict(self, X):
        X = scale(X)
        return [np.argmax(self.prob(x)) for x in X]
