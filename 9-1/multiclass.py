from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model.logistic import LogisticRegression

iris = datasets.load_iris()
X, y = iris.data, iris.target
print OneVsRestClassifier(LogisticRegression(random_state=0)).fit(X, y).predict(X)
