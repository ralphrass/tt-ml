import numpy as np

r = [1, -3, 0, 4, 1, 0, 3]
s = [0, 1, 4, -2, 3, -1, 4]

x = np.array(r)
y = np.array(s)

avg_x = np.sum(x)/float(len(x))
avg_y = np.sum(y)/float(len(y))

print avg_x

cov = sum([(a-avg_x)*(b-avg_y) for a, b in zip(x, y)])

print cov / float(len(x))
