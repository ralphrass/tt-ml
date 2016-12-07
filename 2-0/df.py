# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

data = {'nome': ['Ana', 'Marcia', 'Jose', 'Pedro', 'Paulo', 'Juliana'],
        'enjoo': ['sim', 'nao', 'sim', 'nao', 'nao', 'nao'],
        'diagnostico': ['doente', 'saudavel', 'saudavel', 'doente', 'saudavel', 'doente'],
        'mancha': ['pequena', 'pequena', 'grande', 'pequena', 'grande', 'grande'],
        'dor': ['sim', 'nao', 'nao', 'sim', 'sim', 'sim'],
        'salario': [1000, 1100, 600, 2000, 1800, 900]}
x = pd.DataFrame(data=data)

# print type(x['nome'][0])

x['nome'] = x['nome'].astype('category')
cats = x.select_dtypes(['category']).columns
x[cats] = x[cats].apply(lambda a: a.cat.codes)

# print x

plt.boxplot(x.salario)
# plt.show()

# print np.array(x['nome'].reshape(-1, 1))

# b = preprocessing.Binarizer(threshold=2).fit(x['nome'].reshape(-1, 1))
# print b.transform(x['nome'])

o = preprocessing.OneHotEncoder()
o.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
print o.transform([[0,1,3]])