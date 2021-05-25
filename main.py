import pandas as pd
import numpy as np
import math
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
from pprint import pprint

# A’nın alanı done
# A’nın genişlik/uzunluk oranı done
# A’nın tipi (örn. ikon ya da metin) done
# B’nin alanı done
# B’nin genişlik/uzunluk oranı done
# B’nin tipi (örn. ikon ya da metin) done
# A ile B’nin merkezleri arasındaki çizginin boyu done
# A ile B’nin merkezleri arasındaki çizginin açısı done


dataPath = "./data/kodi-base.csv"
dataOrg = pd.read_csv(dataPath)
data = dataOrg.copy()
data["x"] = data.apply(lambda row: (row.x1 + row.x2) / 2, axis=1)
data["y"] = data.apply(lambda row: (row.y1 + row.y2) / 2, axis=1)
data["w/h"] = data.apply(lambda row: row.width / row.height, axis=1)
data["area"] = data.apply(lambda row: (row.x2 - row.x1) * (row.y2 - row.y1), axis=1)

# data = data.drop(columns=["x1", "y1", "x2", "y2"])
data2 = data

data["key"] = 1
data2["key"] = 1
result = pd.merge(data, data2, on="key").drop("key", 1)
result = result[result["_id_x"] != result["_id_y"]]

result["center_diff"] = result.apply(
    lambda row: math.sqrt(pow((row.x_x - row.x_y), 2) + pow((row.y_x - row.y_y), 2)),
    axis=1,
)
result["center_angle"] = result.apply(
    lambda row: np.rad2deg(np.arctan2((row.y_x - row.y_y), (row.x_x - row.x_y))), axis=1
)
result = result.drop(
    columns=[
        "x1_x",
        "x2_x",
        "y1_x",
        "y2_x",
        "width_x",
        "height_x",
        # "x_x",
        # "y_x",
        "x1_y",
        "x2_y",
        "y1_y",
        "y2_y",
        "width_y",
        "height_y",
        # "x_y",
        # "y_y",
    ]
)

# print(result)
# quit()
numClustersStart = 5
numClustersEnd = 10
step = 1
tol = 1e-04
maxIter = 300
distortions = []

data = result.copy()
data = data.drop(columns=["_id_x", "_id_y"])

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
        dictionary[cluster_labels[i]].append(
            [int(result.iloc[i]["_id_x"]), int(result.iloc[i]["_id_y"])]
        )
    pprint(dictionary)
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
