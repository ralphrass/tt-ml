import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split


# df = pd.read_csv('acidentes-2000.not', sep=';', low_memory=False)
# print len(df)
# print df
# exit()

path = r'../5-2'
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    print file_
    df = pd.read_csv(file_, sep=';', low_memory=False)
    list_.append(df)
frame = pd.concat(list_)

frame.columns = ['ID', 'log1', 'log2', 'predial1', 'local', 'tipo_acidente', 'local_via', 'data_hora', 'dia_sem',
                 'feridos', 'mortes', 'morte_post', 'fatais', 'auto', 'taxi', 'lotacao', 'onibus_urb', 'onibus_int',
                 'caminhao', 'moto', 'carroca', 'bicicleta', 'outro', 'tempo', 'noite_dia', 'fonte', 'boletim',
                 'regiao', 'dia', 'mes', 'ano', 'fx_hora', 'cont_acid', 'cont_vit', 'ups', 'latitude', 'longitude']

# del frame['ID']

frame['mortes'] = frame['mortes'].astype('int')
frame['moto'] = frame['moto'].astype('int')
frame['dia_sem'] = frame['dia_sem'].astype('str')

frame['latitude'] = frame['latitude'].str.replace(',', '.')
frame['longitude'] = frame['longitude'].str.replace(',', '.')

# print frame['latitude']

frame['latitude'] = frame['latitude'].astype('float')
frame['longitude'] = frame['longitude'].astype('float')

# print frame.iloc[1]

X = frame[['latitude', 'longitude']]

# print X

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print kmeans.labels_
print kmeans.cluster_centers_

# print y
# plt.scatter(X['longitude'], X['latitude'], c=['red', 'blue'])
plt.show()
