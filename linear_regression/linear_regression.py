import numpy as np

class LinearRegression():
    def __init__(self, n_epochs=50, lr=0.000002, gradient_descent=False):
        self.n_epochs = n_epochs
        self.lr = lr
        self.gradient_descent = gradient_descent
        
    def predict(self, X):
        return np.matmul(self.add_bias(X), self.w)
    
    def fit(self, X, y):
        return self.fit_with_gradient_descent(X, y) if self.gradient_descent else self.fit_with_normal_equations(X, y)
    
    def fit_with_gradient_descent(self, X, y):
        if y.ndim > 1:
            y = np.squeeze(y)
        X_with_bias = self.add_bias(X)
        self.w = np.zeros((len(X_with_bias[0]),))
        
        for i in range(self.n_epochs):
            # Batch Gradient Descent
            gradient = np.zeros(len(self.w),)
            loss = 0
            
            y_pred = self.predict(X)
            
            for j, x in enumerate(X_with_bias):
                gradient += 2 * (y_pred[j] - y[j]) * x
                
            gradient = gradient / len(y)
            loss = np.mean((y - y_pred) ** 2)
                
            print("Epoch {0}: avg loss is {1:.2f}".format(i, loss))
#             print("Gradient is {}".format(gradient))
#             print("Parameters is {}".format(self.w))

            self.w -= self.lr * gradient
        print("\n Avg loss is: {0:.2f}".format(loss))
    
    def fit_with_normal_equations(self, X, y):
        X = self.add_bias(X)
        self.w = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))
        
        y_pred = np.matmul(X, self.w)
        avg_loss = np.mean((y - y_pred) ** 2)
        print("Avg loss is: {0:.2f}".format(avg_loss))
        
    def add_bias(self, X):
        return np.append(X, np.ones((len(X), 1)), axis=1)