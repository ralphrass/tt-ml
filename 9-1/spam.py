import pandas as pd
df = pd.read_csv('data/SMSSpamCollection', delimiter='\t', header=None)
print df.head()

print 'Nr de mensagens SPAM:', df[df[0] == 'spam'][0].count()
print 'Nr de mensagens Normais:', df[df[0] == 'ham'][0].count()

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score

X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1], df[0])

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# print X_test.shape

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

print enumerate(predictions[:5])

from sklearn.cross_validation import train_test_split, cross_val_score
scores = cross_val_score(classifier, X_train, y_train, cv=5)
print np.mean(scores), scores

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)
