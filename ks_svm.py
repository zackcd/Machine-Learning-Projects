from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utilities import plot_decision_regions


iris = load_iris()
X = iris.data[:, [2,3]]
y = iris.target

test_portion = 0.3
seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_portion, random_state=seed, stratify = y)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

c = 100
svm = SVC(C = 1.0, kernel = 'linear', random_state=seed)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined, classifier = svm, test_idx=range(105, 150))
plt.legend(loc = 'upper left')
plt.show()

