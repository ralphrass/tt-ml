import df as pd
import glob
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
gnb = GaussianNB()

# df = pd.read_csv('acidentes-2000.not', sep=';', low_memory=False)
# print len(df)
# print df
# exit()

path = r'.'
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

print len(frame[frame['dia_sem']=='SEGUNDA-FEIRA'])

# probabilidade de morrer em um acidente de transito em poa?

print frame['local'].unique()

print sum(frame['mortes'])/float(len(frame))

# Qual a probabilidade de haver um acidente envolvendo moto em finais de semana?

print len(frame[frame['dia_sem']=='DOMINGO'])

print len(frame[ ((frame['dia_sem']=='DOMINGO') | (frame['dia_sem']=='SABADO') & (frame['moto'] == 1))])/float(len(frame))

# Treine um modelo considerando que a classe eh o local (logradouro / cruzamento)

categories = frame.select_dtypes(['object']).columns
# print categories
frame['log1'] = frame['log1'].astype('category')
frame['log2'] = frame['log2'].astype('category')
frame['tipo_acidente'] = frame['tipo_acidente'].astype('category')
frame['local_via'] = frame['local_via'].astype('category')
frame['data_hora'] = frame['data_hora'].astype('category')
frame['dia_sem'] = frame['dia_sem'].astype('category')
frame['noite_dia'] = frame['noite_dia'].astype('category')
frame['fonte'] = frame['fonte'].astype('category')
frame['boletim'] = frame['boletim'].astype('category')
frame['regiao'] = frame['regiao'].astype('category')
frame['latitude'] = frame['latitude'].astype('category')
frame['longitude'] = frame['longitude'].astype('category')
frame['tempo'] = frame['tempo'].astype('category')
frame['local'] = frame['local'].astype('category')

categories = frame.select_dtypes(['category']).columns
frame[categories] = frame[categories].apply(lambda x: x.cat.codes)

# print frame.iloc[1]

X = frame[['log1', 'log2', 'predial1', 'tipo_acidente', 'local_via', 'data_hora', 'dia_sem', 'feridos', 'mortes',
           'morte_post', 'fatais', 'auto', 'taxi', 'lotacao', 'onibus_urb', 'onibus_int', 'caminhao', 'moto',
           'carroca', 'bicicleta', 'outro', 'tempo', 'noite_dia', 'fonte', 'boletim', 'regiao', 'dia', 'mes', 'ano',
           'fx_hora', 'cont_acid', 'cont_vit', 'ups', 'latitude', 'longitude']]
y = np.array(frame[['local']])

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# print X_train

gnb.fit(X_train, Y_train.ravel())
predictions = gnb.predict(X_test)

total = sum([p==r for p,r in zip(predictions, Y_test)])

print total / float(len(Y_test))

# print y
