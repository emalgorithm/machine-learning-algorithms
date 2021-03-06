import numpy as np
from scipy.stats import multivariate_normal


class GDA:
    def __init__(self, n_classes, n_features):
        # self.phi contains the per class probabilities
        self.phi = np.zeros(n_classes)
        # self.mu[c] is the mean of the distribution of class c
        self.mu = np.zeros((n_classes, n_features))
        # self.sigma is the shared covariance matrix of the distribution of all classes
        self.sigma = np.zeros((n_features, n_features))

        self.n_classes = n_classes
        self.n_features = n_features

    def fit(self, X, y):
        for c in range(self.n_classes):
            self.phi[c] = len(y[y == c]) / len(y)

            self.mu[c] = np.sum(X[y == c], axis=0) / len(X[y == c])

        # Compute covariance for each matrix and then average them
        self.sigma = sum([np.cov(X[y == c], rowvar=False) for c in range(self.n_classes)]) / \
                     self.n_classes

    def predict(self, X):
        y_pred = np.zeros(len(X))

        for i, x in enumerate(X):
            p = np.zeros(self.n_classes)
            for c in range(self.n_classes):
                p[c] = multivariate_normal.pdf(x, self.mu[c], self.sigma, allow_singular=True)
            y_pred[i] = np.argmax(p)

        return y_pred
