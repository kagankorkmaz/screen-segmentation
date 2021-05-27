import pandas as pd
import numpy as np
import math
import sklearn
from sklearn import preprocessing
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler

# import tensorflow
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from yellowbrick.cluster import SilhouetteVisualizer
import os
from collections import Counter, defaultdict, OrderedDict

# A’nın alanı done
# A’nın genişlik/uzunluk oranı done
# A’nın tipi (örn. ikon ya da metin) done
# B’nin alanı done
# B’nin genişlik/uzunluk oranı done
# B’nin tipi (örn. ikon ya da metin) done
# A ile B’nin merkezleri arasındaki çizginin boyu done
# A ile B’nin merkezleri arasındaki çizginin açısı done

# ayni columnda olanlar absolute(86-94)
# ayni row olanlar abs(177 - 183)


dataPath = "./data/kodi-base.csv"
dataOrg = pd.read_csv(dataPath)
data = dataOrg.copy()
# print(data)
# data["x"] = data.apply(lambda row: (row.x1 + row.x2) / 2, axis=1)
data["y"] = data.apply(lambda row: (row.y1 + row.y2) / 2, axis=1)  # (x1,y)
# data["w/h"] = data.apply(lambda row: row.width / row.height, axis=1)
# data["area"] = data.apply(lambda row: (row.x2-row.x1)
#                          * (row.y2 - row.y1), axis=1)
# data = data.drop(columns=["x1", "y1", "x2", "y2"])
data2 = data

data["key"] = 1
data2["key"] = 1
result = pd.merge(data, data2, on="key").drop("key", 1)
result = result[result["_id_x"] < result["_id_y"]]

result["center_diff"] = result.apply(
    lambda row: math.sqrt(pow((row.x1_x - row.x1_y), 2) + pow((row.y_x - row.y_y), 2)),
    axis=1,
)
result["center_angle"] = result.apply(
    lambda row: np.rad2deg(np.arctan2((row.y_x - row.y_y), (row.x1_x - row.x1_y))),
    axis=1,
)
result = result.drop(
    columns=[
        "x2_x",
        "y1_x",
        "y2_x",
        "width_x",
        "height_x",
        "x1_y",
        "x2_y",
        "y1_y",
        "y2_y",
        "width_y",
        "height_y",
    ]
)
result["same_row"] = result.apply(
    lambda row: "1"
    if abs(row.center_angle) > 177 and abs(row.center_angle) < 183
    else "0",
    axis=1,
)
result["same_column"] = result.apply(
    lambda row: "1"
    if abs(row.center_angle) > 86 and abs(row.center_angle) < 94
    else "0",
    axis=1,
)
result["ndist"] = MinMaxScaler().fit_transform(
    np.array(result["center_diff"]).reshape(-1, 1)
)

##########################################################################
# y value is the middle point between y1 and y2 values
# x values is the middle point between x1 and x2 values
# _x represents first box values and _y represents second box values
# center_diff, the distance between (x1_x, y_x) and (x1_y, y_y) points.
# center_angle, the angle between (x1_x, y_x) and (x1_y, y_y) points.
# inside value is 1 if box contains an icon, 0 if box contains a label
# same_row is 1 if the absolute value of the angle between the points (x1_x, y_x) and (x1_y, y_y) is between 177 and 183, else same_row is 0
# same_column is 1 if the absolute value of the angle between the points (x1_x, y_x) and (x1_y, y_y) is between 86 and 94, else same_row is 0
# ndist is the normalized values of center_diff values, MinMaxScaler() function is used.
# normalizing the distance between bounding box may increase the clustering performance
############################################################################
# result = result.drop(columns=['x1_x', 'inside_x', 'y_x', 'inside_y', 'y_y',
#        'center_diff', 'center_angle', 'same_row', 'same_column', 'ndist'])
only_id = result.drop(
    columns=[
        "x1_x",
        "inside_x",
        "y_x",
        "inside_y",
        "y_y",
        "center_diff",
        "center_angle",
        "same_row",
        "same_column",
        "ndist",
    ]
)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(result)
result = result.drop(
    columns=["x1_x", "_id_x", "y_x", "_id_y", "y_y", "center_diff", "center_angle"]
)
# print(result)

clustering = DBSCAN(eps=0.5, min_samples=2).fit(result)
# print(clustering.labels_)

only_id["labels"] = clustering.labels_
clusters = [None] * len(set(clustering.labels_))
for x in range(len(set(clustering.labels_))):
    clusters[x] = []


for i in range(len(only_id)):
    clusters[only_id["labels"].values[i]].append([only_id["_id_x"].values[i], only_id["_id_y"].values[i]])

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(only_id)

for i in range(len(clusters)):
    print(i)
    pprint(clusters[i])
    print("--------------------------------")
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
