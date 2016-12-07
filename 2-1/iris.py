import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

iris = pd.read_csv('iris.data')
iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# print iris.iloc[0:3, 0:2]
x = np.array(iris['sepal_width']).reshape(-1, 1)
y = np.array(iris['sepal_length']).reshape(-1, 1)

# print x

# exit()

plt.figure()
plt.boxplot([iris['sepal_length'], iris['sepal_width']], showmeans=True)
plt.show()
