from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):
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
            for x_i, y_i in zip(X, y):
                pred = self.predict(x_i)
                gradient = self.lambd * (y_i - pred)
                self.w[1:] += gradient * x_i
                self.w[0] += gradient

        return self

    def predict(self, x):
        pred = np.dot(x, self.w[1:]) + self.w[0]
        if pred >= 0:
            return 1
        else:
            return -1

class Adaline(object):
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
        pred = np.dot(x, self.w[1:]) + self.w[0]
        if pred >= 0:
            return 1
        else:
            return -1

class SGD(object):
    pass


iris = load_iris()
X = np.asarray(iris.data[:100, [0, 2]])
y = np.asarray(iris.target[:100])
y = [i if i == 1 else -1 for i in y]

plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'setosa')
plt.scatter(X[50:, 0], X[50:, 1], color = 'red', marker = 'x', label = 'versicolor')

plt.xlabel("Sepal length (cm)")
plt.ylabel("Petal length (cm)")
plt.legend(loc = "upper left")
plt.show()

ppn = Perceptron(0.01, 20)
ppn.train(X, y)

ada = Adaline(0.01, 20)
ada.train(X, y)