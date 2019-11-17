from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(X, y, classifier, test_idx = None, resolution = 0.02):
    labels = np.unique(y)
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'grey', 'cyan')
    cmap = ListedColormap(colors[:len(labels)])

    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1

    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(labels):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha = 0.8, c = colors[idx], marker = markers[idx], label = cl, edgecolor = 'black')

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
         
        plt.scatter(X_test[:, 0], X_test[:, 1], c = '', edgecolor = 'black', alpha = 1.0, linewidth = 1, marker = 'o', s = 100, label = 'test set')

def get_accuracy(prediction, target):
    assert len(prediction) == len(target)
    accuracy = accuracy_score(prediction, target)
    return accuracy

def get_mse(prediction, target):
    assert len(prediction) == len(target)
    n = len(prediction)
    dif = (prediction - target) ** 2
    s = dif.sum()
    return s / n