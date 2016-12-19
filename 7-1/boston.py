import numpy as np
from sklearn.datasets import load_boston
boston = load_boston()

from matplotlib import pyplot as plt
plt.scatter(boston.data[:,5], boston.target, color='r')

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

x = boston.data[:,5]
y = boston.target
x = np.transpose(np.atleast_2d(x))
lr.fit(x, y)
y_predicted = lr.predict(x)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y, lr.predict(x))
print("Mean squared error (of training data): {:.3}".format(mse))

x = boston.data
y = boston.target
lr.fit(x, y)

p = lr.predict(x)
plt.scatter(p, y)
plt.xlabel('Predicted price')
plt.ylabel('Actual price')
plt.plot([y.min(), y.max()], [[y.min()], [y.max()]])

plt.show()
