# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

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

print x