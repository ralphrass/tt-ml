import pandas as pd
import numpy as np
import random

df = pd.read_csv('wholesale.csv')

centroides = {'c0', 'c1'}
clusters, clusters_instancias = {}, {}

for c in centroides:
    clusters[c] = np.array([random.uniform(np.min(df[col]), np.max(df[col])) for col in df.columns])
    clusters_instancias[c] = {}

def euclidean(x, y):
    r = np.sqrt(np.sum([(a - b) ** 2 for a, b in zip(x, y)]))
    return r

tamanhos_centroides_anterior = np.array([])
tamanhos_centroides_atual = np.array([])

SAFE_EXIT = 20
count = 0

while True:

    tamanhos_clusters_anterior = [len(clusters_instancias[c]) for c in centroides]

    # zera os clusters
    for c in centroides:
        clusters_instancias[c] = []

    # popula os centroides
    for i in range(0, len(df)):
        instancia = df.iloc[i, :].values
        distancias = [euclidean(np.array(instancia), clusters[c]) for c in clusters]
        clusters_instancias['c' + str(distancias.index(min(distancias)))].append(instancia)

    tamanhos_clusters_atual = [len(clusters_instancias[c]) for c in centroides]

    # move os centroides
    for chave, valor in clusters.iteritems():
        clusters[chave] = np.average(clusters_instancias[chave], axis=0)

    diff = sum([int(a != b) for a, b in zip(sorted(tamanhos_clusters_anterior), sorted(tamanhos_clusters_atual))])
    # print sorted(tamanhos_clusters_anterior), sorted(tamanhos_clusters_atual), diff
    if diff == 0:
        break

    count += 1

    if count % SAFE_EXIT == 0:
        print "Safe exiting"
        break
