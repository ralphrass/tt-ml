import pandas as pd
import numpy as np
iris = pd.read_csv('../2-1/iris.data')
iris.columns = ['sl', 'sw', 'pl', 'pw', 'c']

x = np.array([6.1,2.1,4.7,1.3])
y = np.array([6.6,3.1,5.7,2.1])

k = 3

def euclidean(x, y):
    r = np.sqrt(np.sum([(a-b)**2 for a, b in zip(x, y)]))
    return r

dist_x = []

for idx, linha in iris.iterrows():
    h = np.array(linha[0:4])
    dist_x.append((idx, euclidean(h, y)))

xx = [iris.iloc[idx]['c'] for idx, dist in sorted(dist_x, key=lambda a: a[1])][:k]

d = []

for classe in iris['c'].unique():
    d.append((classe, xx.count(classe)))

d = sorted(d, key=lambda a: a[1], reverse=True)

print d[0][0]
