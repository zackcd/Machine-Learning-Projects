from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression(object):
    def __init__(self, learningRate, iterations, seed = 1):
        self.lambd = learningRate
        self.iter = iterations
        self.w = None

    def train(self, X, y):
        n, m = X.shape
        leny = len(y)
        assert n == leny

        self.w = np.zeros(m + 1)

        self.predict([1, 1])

        for _ in range(self.iter):
            pred = np.dot(X, self.w[1:]) + self.w[0]
            error = (y - pred)
            self.w[1:] += self.lambd * X.T.dot(error)
            self.w[0] += self.lambd * sum(error)

        return self

    def predict(self, x):
        pred = self.sigmoid(x)
        return np.where(pred >= 0.5, 1, 0)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))


iris = load_iris()
X = np.asarray(iris.data[:100, [0, 2]])
y = np.asarray(iris.target[:100])
y = [i if i == 1 else -1 for i in y]

ada = LogisticRegression(0.01, 20)
ada.train(X, y)