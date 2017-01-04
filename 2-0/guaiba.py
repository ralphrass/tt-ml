import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_json('guaiba.json')

print df.columns

temperatura = df['temperatura'].replace(to_replace='', value=np.nan)

imp = Imputer(missing_values='NaN', strategy='mean', axis=1)
imp.fit([temperatura])
temp_trans = imp.transform([temperatura])
print np.std(temp_trans)
X_sc = preprocessing.scale(temp_trans)
print X_sc.std()

imp.fit([df['altura']])
altura_trans = preprocessing.scale([df['altura']])
print np.std(df['altura'])
print altura_trans.std()

plt.hist(df['altura'])
plt.show()