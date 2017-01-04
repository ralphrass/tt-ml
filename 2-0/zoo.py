import pandas as pd
import random
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('zoo.csv')

X = df.iloc[:, 0:len(df.columns)-1]
y = df.iloc[:, len(df.columns)-1]
del X['animal_name']
#
# print y
# exit()

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.7)
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
preds = knn.predict(X_test)

acuracia = sum([1 if p == r else 0 for p, r in zip(preds, Y_test)])

print acuracia / float(len(Y_test))

acuracia_randomico = sum([1 if p == r else 0 for p, r in zip([random.randrange(1, 8) for x in range(len(Y_test))], Y_test)])
print acuracia_randomico / float(len(Y_test))
