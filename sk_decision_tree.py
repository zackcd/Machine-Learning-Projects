from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utilities import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

test_portion = 0.3
seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_portion, random_state = seed, stratify = y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

depth = 4
tree = DecisionTreeClassifier(criterion='gini', max_depth = depth, random_state = seed)
tree.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier = tree, test_idx=range(105, 150))
plt.legend(loc = 'upper left')
plt.show()