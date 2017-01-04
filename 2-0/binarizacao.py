from sklearn import preprocessing

Y = [[ 1., 2.]]
X = [[ 1., -1.,  2.], [ 2.,  0.,  0.], [ 0.,  1., -1.]]

bins = preprocessing.Binarizer(threshold=1.2).fit(Y)

# print bins.transform(Y)

enc = preprocessing.OneHotEncoder()
enc.fit([[0], [1], [2], [1]])
print enc.transform([[0], [1], [2]]).toarray()
