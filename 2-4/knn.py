import random
import df as pd
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

base = pd.read_csv('spambase.data', header=None)
# base.columns()
# print base
# print base.isnull().values.any()

X = base.iloc[:, :56].values
y = base.iloc[:, 57].values

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=0)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)

random_predictions = list(random.randint(0, 1) for i in range(0, len(predictions)))

total = sum(p == r for p, r in zip(predictions, Y_test))
total_random = sum(p == r for p, r in zip(random_predictions, Y_test))

print total/float(len(predictions))
print total_random/float(len(predictions))

mms = MinMaxScaler()
std = StandardScaler()

X_train_mms = mms.fit_transform(X_train)
X_test_mms = mms.transform(X_test)

knn.fit(X_train_mms, Y_train)
predictions = knn.predict(X_test_mms)
total = sum(p == r for p, r in zip(predictions, Y_test))
print total/float(len(predictions))

X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)

knn.fit(X_train_std, Y_train)
predictions = knn.predict(X_test_std)
total = sum(p == r for p, r in zip(predictions, Y_test))
print total/float(len(predictions))


