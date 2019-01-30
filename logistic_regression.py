import numpy as np
from scipy.special import expit


class LogisticRegression:
    def __init__(self, n_features, n_epochs=100):
        # Account for bias
        self.w = np.zeros(n_features + 1)
        self.lr = 0.005
        self.n_epochs = n_epochs

    def prob(self, x):
        """Probability that x belongs to the first class"""
        return expit(np.dot(self.w, x))

    @staticmethod
    def add_bias(X):
        return np.append(X, np.ones((len(X), 1)), axis=1)

    def fit(self, X, y):
        X_with_bias = self.add_bias(X)

        for k in range(self.n_epochs):
            ps = expit(np.matmul(X_with_bias, self.w))
            gradient = sum([(y[i] - ps[i]) * x for i, x in enumerate(X_with_bias)])

            # Compute log likelihood
            ps[y == 0] = 1 - ps[y == 0]
            log_likelihood = sum(ps)

            # Don't divide gradient by number of examples otherwise step becomes too small
            # Batch Gradient Descent
            self.w += self.lr * gradient

            avg_log_likelihood = log_likelihood / len(X)
            print("Epoch {}, avg log likelihood: {}".format(k, avg_log_likelihood))

    def predict(self, X):
        return [1 if self.prob(x) >= 0.5 else 0 for x in self.add_bias(X)]
