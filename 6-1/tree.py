import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import Imputer

base = pd.read_csv('adult.data', header=None, na_values=["?"])
base_test = pd.read_csv('adult.test', header=None, na_values=["?"])

base.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'class']

base_test.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                     'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                     'class']

# base['workclass'].replace('?', np.NaN)

base['sex'] = base['sex'].astype('category')
base['workclass'] = base['workclass'].astype('category')
base['education'] = base['education'].astype('category')
base['marital-status'] = base['marital-status'].astype('category')
base['occupation'] = base['occupation'].astype('category')
base['relationship'] = base['relationship'].astype('category')
base['race'] = base['race'].astype('category')
base['class'] = base['class'].astype('category')
base['native-country'] = base['native-country'].astype('category')

base_test['sex'] = base_test['sex'].astype('category')
base_test['workclass'] = base_test['workclass'].astype('category')
base_test['education'] = base_test['education'].astype('category')
base_test['marital-status'] = base_test['marital-status'].astype('category')
base_test['occupation'] = base_test['occupation'].astype('category')
base_test['relationship'] = base_test['relationship'].astype('category')
base_test['race'] = base_test['race'].astype('category')
base_test['class'] = base_test['class'].astype('category')
base_test['native-country'] = base_test['native-country'].astype('category')

categories = base.select_dtypes(['category']).columns
base[categories] = base[categories].apply(lambda x: x.cat.codes)

categories = base_test.select_dtypes(['category']).columns
base_test[categories] = base_test[categories].apply(lambda x: x.cat.codes)
# print categories
# print base

X = base.iloc[:, :13]
Y = base.iloc[:, 14]

# print X['sex']

X_test = base_test.iloc[:, :13]
Y_test = base_test.iloc[:, 14]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
predictions = clf.predict(X_test)

total = sum(p == r for p, r in zip(predictions, Y_test))
print total / float(len(predictions))

print base.isnan().values.any()


imr = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imr.fit(base)
imr.transform(base_test)
# imputed_data = imr.transform(base.values)
# print imputed_data

clf = clf.fit(base, Y)
# imputed_data_test = imr.transform(base_test.values)
predictions = clf.predict(base_test)

# print base

total = sum(p == r for p, r in zip(predictions, Y_test))
print total / float(len(predictions))
