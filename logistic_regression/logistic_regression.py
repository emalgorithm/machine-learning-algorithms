import numpy as np
from scipy.special import expit
from sklearn.preprocessing import scale


class LogisticRegression:
    def __init__(self, n_features, n_epochs=100):
        # Account for bias
        self.w = np.random.normal(0, 1, n_features + 1)
        self.lr = 0.05
        self.n_epochs = n_epochs

    def prob(self, x):
        """Probability that x belongs to the first class"""
        return expit(np.dot(self.w, x))

    @staticmethod
    def add_bias(X):
        return np.append(X, np.ones((len(X), 1)), axis=1)

    def fit(self, X, y, verbose=True):
        X_with_bias = self.add_bias(X)
        X_with_bias = scale(X_with_bias)

        for k in range(self.n_epochs):
            ps = expit(np.matmul(X_with_bias, self.w))
            # Compute gradient of log likelihood
            gradient = np.mean([(y[i] - ps[i]) * x for i, x in enumerate(X_with_bias)], 0)

            # Batch Gradient Ascent
            # Add gradient since we are maximizing log likelihood
            self.w += self.lr * gradient

            if verbose:
                # Compute log likelihood
                ps[y == 0] = 1 - ps[y == 0]
                likelihood = np.mean(ps)
                print("Epoch {}, likelihood: {}".format(k, likelihood))

    def predict(self, X):
        X_with_bias = scale(self.add_bias(X))
        return [1 if self.prob(x) >= 0.5 else 0 for x in X_with_bias]
