import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
gnb = GaussianNB()

df = pd.read_csv('mammographic_masses.data', na_values=['?'])
df.columns = ['a', 'b', 'c', 'd', 'e', 'f']
df.fillna(np.mean(df), inplace=True)

# print df['a']

X = df.iloc[:, :4]
Y = df.iloc[:, 5]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=0)

# print "X",X_train
# print "Y",Y_train

gnb.fit(X_train, Y_train)

predictions = gnb.predict(X_test)
total = sum(p == r for p, r in zip(predictions, Y_test))

print total/float(len(predictions))
