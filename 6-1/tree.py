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

# print base.dtypes

objects = base.select_dtypes(['object']).columns

for obj in objects:
    base[obj] = base[obj].astype('category')

for obj in objects:
    base_test[obj] = base_test[obj].astype('category')

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

# print base.isnan().values.any()


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
