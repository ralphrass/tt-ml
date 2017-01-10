import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
# from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('RH.csv')

X = df.loc[:, ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company',
               'Work_accident', 'promotion_last_5years', 'sales', 'salary']]
Y = df.loc[:, 'left']
print Y.head()

del X['sales']
del X['salary']
del X['satisfaction_level']
del X['last_evaluation']
del X['number_project']
del X['time_spend_company']

# skf = StratifiedKFold(n_folds=5, random_state=0, y=Y)
skf = KFold(len(Y), n_folds=5)

clf = tree.DecisionTreeClassifier()
knn = KNeighborsClassifier(n_neighbors=10)

s = 0
sk = 0

for treino, teste in skf:

    # print treino, len(treino), type(treino), list(treino)
    X_treino = X.iloc[treino, :]
    X_teste = X.iloc[teste, :]
    Y_teste = Y[teste]
    Y_treino = Y[treino]

    clf.fit(X_treino, Y_treino)
    knn.fit(X_treino, Y_treino)

    predictions = clf.predict(X_teste)
    preds_knn = knn.predict(X_teste)

    print accuracy_score(Y_teste, predictions)
    print accuracy_score(Y_teste, preds_knn), "kNN"
    s += accuracy_score(Y_teste, predictions)
    sk += accuracy_score(Y_teste, preds_knn)
    # print confusion_matrix(Y_teste, predictions)

print s / 5., "is the average"
print sk / 5., "is the average for kNN"