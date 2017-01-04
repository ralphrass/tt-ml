import pandas as pd

from sklearn import preprocessing
import numpy as np

df = pd.read_csv('acidentes-2005.csv', sep=';', dtype=object)

print df['TIPO_ACID'].unique()

df['TIPO_ACID'] = df['TIPO_ACID'].astype('category')
df['DIA_SEM'] = df['DIA_SEM'].astype('category')
df['LOCAL_VIA'] = df['LOCAL_VIA'].astype('category')
categ = df.select_dtypes(['category']).columns

df[categ] = df[categ].apply(lambda x: x.cat.codes)

print df['TIPO_ACID'].unique()

res = pd.get_dummies(df['TIPO_ACID'])
print res

# enc = preprocessing.OneHotEncoder()
# enc.fit([df['TIPO_ACID'].head()])
# print enc.transform([df['TIPO_ACID'].head()]).toarray()

