import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data[:, [2,3]]
y = iris.target

class_labels = np.unique(y)

test_portion = 0.3
seed = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_portion, random_state=seed, stratify = y)

# print(np.bincount(y))
# print(np.bincount(y_test))
# print(np.bincount(y_train))

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

iter = 100
learningRate = 0.01
ppn = Perceptron(max_iter = iter, eta0 = learningRate, random_state = seed)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)

error = (y_test != y_pred).sum()
print(error)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)



