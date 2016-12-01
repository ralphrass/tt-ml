import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

std = StandardScaler()

df = pd.read_table('housing.data', delim_whitespace=True)

X = df.iloc[:, :12].values
y = df.iloc[:, 13].values

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.4, random_state=0)

clf = linear_model.LinearRegression(normalize=True)
clf.fit(X_train, Y_train)

predictions = clf.predict(X_test)
all = zip(Y_test, predictions)

print X_train

diff = sum([abs(p-r) for p, r in all])
print diff

X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)

print X_train_std

clf.fit(X_train_std, Y_train)

predictions = clf.predict(X_test_std)
all = zip(Y_test, predictions)

diff = sum([abs(p-r) for p, r in all])
print diff

