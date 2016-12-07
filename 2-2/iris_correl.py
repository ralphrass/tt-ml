import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

df = pd.read_csv('../2-1/iris.data')
df.columns = ['sl', 'sw', 'pl', 'pw', 'c']

df['c'] = df['c'].astype('category')
cats = df.select_dtypes(['category']).columns
df[cats] = df[cats].apply(lambda x: x.cat.codes)

# print df

plt.scatter(df['sl'], df['sw'], c=df['c'])
# plt.show()

x_scaled = preprocessing.scale(df['sl'])

scaler = preprocessing.StandardScaler().fit(np.array(df['sl']))

print scaler.transform(df['sl'])

# print x_scaled.mean(axis=0)