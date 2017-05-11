import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import Imputer

base = pd.read_csv('adult.data', header=None, na_values=['?'])
base_test = pd.read_csv('adult.test', header=None, na_values=['?'])

# base = pd.read_csv('adult_NaN.data', header=None, na_values='NaN')
# base_test = pd.read_csv('adult.test', header=None, na_values='NaN')

# print base_test

base.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'class']
base['age'] = base['age'].astype(float)
base['workclass'] = base['workclass'].astype('category')
base['fnlwgt'] = base['fnlwgt'].astype(float)
base['education'] = base['education'].astype('category')
base['education-num'] = base['education-num'].astype(float)
base['marital-status'] = base['marital-status'].astype('category')
base['occupation'] = base['occupation'].astype('category')
base['relationship'] = base['relationship'].astype('category')
base['race'] = base['race'].astype('category')
base['sex'] = base['sex'].astype('category')
base['capital-gain'] = base['capital-gain'].astype(float)
base['capital-loss'] = base['capital-loss'].astype(float)
base['hours-per-week'] = base['hours-per-week'].astype(float)
base['native-country'] = base['native-country'].astype('category')
base['class'] = base['class'].astype('category')

base_test.columns = base.columns

base_test.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                'class']
base_test['age'] = base_test['age'].astype(float)
base_test['workclass'] = base_test['workclass'].astype('category')
base_test['fnlwgt'] = base_test['fnlwgt'].astype(float)
base_test['education'] = base_test['education'].astype('category')
base_test['education-num'] = base_test['education-num'].astype(float)
base_test['marital-status'] = base_test['marital-status'].astype('category')
base_test['occupation'] = base_test['occupation'].astype('category')
base_test['relationship'] = base_test['relationship'].astype('category')
base_test['race'] = base_test['race'].astype('category')
base_test['sex'] = base_test['sex'].astype('category')
base_test['capital-gain'] = base_test['capital-gain'].astype(float)
base_test['capital-loss'] = base_test['capital-loss'].astype(float)
base_test['hours-per-week'] = base_test['hours-per-week'].astype(float)
base_test['native-country'] = base_test['native-country'].astype('category')
base_test['class'] = base_test['class'].astype('category')

for c in base.columns:
    print c, base[c].dtype

objects = base.select_dtypes(['object']).columns

for obj in objects:
    base[obj] = base[obj].astype('category')

for obj in objects:
    base_test[obj] = base_test[obj].astype('category')


print base.iloc[106, :]

print base['workclass'].describe()
# print base.replace('?', np.nan)
# base = base.fillna(base.mean(), inplace=True)
# print base.iloc[106, :]

categories = base.select_dtypes(['category']).columns
base[categories] = base[categories].apply(lambda x: x.cat.codes)

categories = base_test.select_dtypes(['category']).columns
base_test[categories] = base_test[categories].apply(lambda x: x.cat.codes)

# exit()

# print categories
# print base

X = base.iloc[:, :13]
Y = base.iloc[:, 14]

# print X['sex']

X_test = base_test.iloc[:, :13]
Y_test = base_test.iloc[:, 14]

clf = tree.DecisionTreeClassifier(random_state=0)
clf = clf.fit(X, Y)
predictions = clf.predict(X_test)

total = sum(p == r for p, r in zip(predictions, Y_test))
print total / float(len(predictions))

# imputed_data = imr.transform(base.values)
# print imputed_data

# clf = clf.fit(X, Y)
# imputed_data_test = imr.transform(base_test.values)
# predictions = clf.predict(X_test)

# print base

# total = sum(p == r for p, r in zip(predictions, Y_test))
# print total / float(len(predictions))
