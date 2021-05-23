import pandas as pd
import numpy as np
import sklearn

# import tensorflow
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from yellowbrick.cluster import SilhouetteVisualizer
import os
from collections import Counter, defaultdict, OrderedDict

dataPath = "./data/kodi-base.csv"
dataOrg = pd.read_csv(dataPath)
data = dataOrg.copy()
data["x"] = data.apply(lambda row: (row.x1 + row.x2) / 2, axis=1)
data["y"] = data.apply(lambda row: (row.y1 + row.y2) / 2, axis=1)
data["w/h"] = data.apply(lambda row: row.width / row.height, axis=1)
data = data.drop(columns=["x1", "y1", "x2", "y2"])
data2 = data

data['key'] = 1
data2['key'] = 1
result = pd.merge(data,data2,on='key').drop('key',1)
result = result[result['_id_x'] != result['_id_y']]
print(result)
quit()
joined = data.join(data, lsuffix="_x", rsuffix="_y")
print("Joined:\n", joined)
print("data:\n" ,data)
quit()
numClustersStart = 2
numClustersEnd = 5
step = 1
tol = 1e-04
maxIter = 300
distortions = []

for n_clusters in range(numClustersStart, numClustersEnd, step):
    km = KMeans(
        n_clusters=n_clusters, init="k-means++", n_init=10, max_iter=maxIter, tol=tol
    )
    visualizer = SilhouetteVisualizer(km, colors="yellowbrick")
    visualizer.fit(np.array(data))
    km.fit(data)
    distortions.append(km.inertia_)
    cluster_labels = km.predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)

    opath = str(1)
    path = os.path.join("./", opath)
    if not os.path.exists(path):
        os.mkdir(path)
    startEndPath = str(numClustersStart) + "-" + str(numClustersEnd)
    visualizer.show(outpath="./" + opath + "/" + str(n_clusters) + ".png")  # TODO
    plt.cla()
    plt.clf()
    plt.close("all")
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )
    print("For n_clusters =", n_clusters, "The distortion is :", km.inertia_)
    dictionary = {}
    print(n_clusters, "clusters:")
    for i in range(n_clusters):
        dictionary[i] = []
    for i in range(len(cluster_labels)):
        dictionary[cluster_labels[i]].append(dataOrg.loc[i, "_id"])
    print(dictionary)
    print("------------------------------\n")


plt.cla()
plt.clf()
plt.close("all")
plt.plot(range(numClustersStart, numClustersEnd, step), distortions, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")
# plt.show()
plt.savefig("./" + opath + "/" + startEndPath + "-distortion")
plt.close("all")