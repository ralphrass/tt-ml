import pandas as pd
import numpy as np
import math

df = pd.read_csv('pesos_alturas_english.csv')

print df.columns

altura = df.loc[:, 'Altura']
peso = df.loc[:, 'Peso']

# print np.corrcoef(altura, peso)[0, 1]
# print np.dot(altura, peso) / (np.linalg.norm(altura) * np.linalg.norm(peso))

x = [3, 2, 0, 5, 0, 0, 0, 2, 0, 0]
y = [1, 0, 0, 0, 0, 0, 0, 1, 0, 2]


print np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

print np.sum(np.cov(x, y)) / (np.std(x)**2 * np.std(y)**2)

print math.sqrt(sum([(a-p)**2 for a, p in zip(x, y)]))
