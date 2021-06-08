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
data["x"] = data.apply(lambda row: (row.x1 + row.x2) / 2, axis=1)
data["y"] = data.apply(lambda row: (row.y1 + row.y2) / 2, axis=1)  # (x1,y)


# data["w/h"] = data.apply(lambda row: row.width / row.height, axis=1)
# data["area"] = data.apply(lambda row: (row.x2-row.x1)
#                          * (row.y2 - row.y1), axis=1)
# data = data.drop(columns=["x1", "y1", "x2", "y2"])


data["leftUpCorner"] = data.apply(lambda row: [row.x1, row.y1], axis=1)
data["rightUpCorner"] = data.apply(lambda row: [row.x1 + row.width, row.y1], axis=1)
data["leftBottomCorner"] = data.apply(lambda row: [row.x1, row.y1 + row.height], axis=1)
data["rightBottomCorner"] = data.apply(
    lambda row: [row.x1 + row.width, row.y1 + row.height], axis=1
)
corners = data.copy()
corners = corners.drop(
    columns=["x1", "x2", "y1", "y2", "inside", "width", "height", "_id", "x", "y"]
)


data["topEdgeCenter"] = data.apply(lambda row: [row.x1 + row.width / 2, row.y1], axis=1)
data["BottomEdgeCenter"] = data.apply(
    lambda row: [row.x1 + row.width / 2, row.y1 + row.height], axis=1
)
data["leftEdgeCenter"] = data.apply(
    lambda row: [row.x1, row.y1 + row.height / 2], axis=1
)
data["rightEdgeCenter"] = data.apply(
    lambda row: [row.x1 + row.width, row.y1 + row.height / 2], axis=1
)
data["center"] = data.apply(
    lambda row: [row.x1 + row.width / 2, row.y1 + row.height / 2], axis=1
)

data = data.drop(columns=["width", "height"])

data2 = data.copy()
data["key"] = 1
data2["key"] = 1

result = pd.merge(data, data2, on="key").drop("key", 1)
result = result[result["_id_x"] < result["_id_y"]]

result["center_angle"] = result.apply(
    lambda row: np.rad2deg(np.arctan2((row.y_x - row.y_y), (row.x1_x - row.x1_y))),
    axis=1,
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
sameRowColumn = result.copy()

sameRowColumn = sameRowColumn.drop(
    columns=[
        "x1_x",
        "x2_x",
        "y1_x",
        "y2_x",
        "inside_x",
        "x_x",
        "y_x",
        "leftUpCorner_x",
        "rightUpCorner_x",
        "leftBottomCorner_x",
        "rightBottomCorner_x",
        "topEdgeCenter_x",
        "BottomEdgeCenter_x",
        "leftEdgeCenter_x",
        "rightEdgeCenter_x",
        "center_x",
        "x1_y",
        "x2_y",
        "y1_y",
        "y2_y",
        "inside_y",
        "x_y",
        "y_y",
        "leftUpCorner_y",
        "rightUpCorner_y",
        "leftBottomCorner_y",
        "rightBottomCorner_y",
        "topEdgeCenter_y",
        "BottomEdgeCenter_y",
        "leftEdgeCenter_y",
        "rightEdgeCenter_y",
        "center_y",
        "center_angle",
    ]
)


only_id = result.drop(
    columns=[
        "leftUpCorner_x",
        "rightUpCorner_x",
        "leftBottomCorner_x",
        "rightBottomCorner_x",
        "topEdgeCenter_x",
        "BottomEdgeCenter_x",
        "leftEdgeCenter_x",
        "rightEdgeCenter_x",
        "center_x",
        "leftUpCorner_y",
        "rightUpCorner_y",
        "leftBottomCorner_y",
        "rightBottomCorner_y",
        "topEdgeCenter_y",
        "BottomEdgeCenter_y",
        "leftEdgeCenter_y",
        "rightEdgeCenter_y",
        "center_y",
        "x_x",
        "x_y",
        "y_x",
        "y_y",
        "center_angle",
        "inside_x",
        "inside_y",
        "x1_x",
        "x2_x",
        "y1_x",
        "y2_x",
        "x1_y",
        "x2_y",
        "y1_y",
        "y2_y",
    ]
)


for a in range(9):
    first = ""
    first_lst = [
        "leftUpCorner_x",
        "rightUpCorner_x",
        "leftBottomCorner_x",
        "rightBottomCorner_x",
        "topEdgeCenter_x",
        "BottomEdgeCenter_x",
        "leftEdgeCenter_x",
        "rightEdgeCenter_x",
        "center_x",
    ]
    first = first_lst[a]

    for b in range(9):
        column_index = a * 9 + b
        column_name = "diff_" + str(column_index)
        angle_name = "angle_" + str(column_index)

        second = ""
        second_lst = [
            "leftUpCorner_y",
            "rightUpCorner_y",
            "leftBottomCorner_y",
            "rightBottomCorner_y",
            "topEdgeCenter_y",
            "BottomEdgeCenter_y",
            "leftEdgeCenter_y",
            "rightEdgeCenter_y",
            "center_y",
        ]
        second = second_lst[b]

        ls = []
        ls_angle = []
        for c in range(len(result.index)):
            ls.append(
                math.sqrt(
                    pow((result[first].iloc[c][0] - result[second].iloc[c][0]), 2)
                    + pow((result[first].iloc[c][1] - result[second].iloc[c][1]), 2)
                )
            )
            ls_angle.append(
                np.rad2deg(
                    np.arctan2(
                        (result[first].iloc[c][1] - result[second].iloc[c][1]),
                        (result[first].iloc[c][0] - result[second].iloc[c][0]),
                    )
                )
            )
        result[column_name] = ls
        result[angle_name] = ls_angle
        result[column_name] = MinMaxScaler().fit_transform(
            np.array(result[column_name]).reshape(-1, 1)
        )
        result[angle_name] = MinMaxScaler().fit_transform(
            np.array(result[angle_name]).reshape(-1, 1)
        )

result = result.drop(
    columns=[
        "_id_x",
        "_id_y",
        "leftUpCorner_x",
        "rightUpCorner_x",
        "leftBottomCorner_x",
        "rightBottomCorner_x",
        "topEdgeCenter_x",
        "BottomEdgeCenter_x",
        "leftEdgeCenter_x",
        "rightEdgeCenter_x",
        "center_x",
        "leftUpCorner_y",
        "rightUpCorner_y",
        "leftBottomCorner_y",
        "rightBottomCorner_y",
        "topEdgeCenter_y",
        "BottomEdgeCenter_y",
        "leftEdgeCenter_y",
        "rightEdgeCenter_y",
        "center_y",
        "x_x",
        "x_y",
        "y_x",
        "y_y",
        "center_angle",
        "x1_x",
        "x2_x",
        "y1_x",
        "y2_x",
        "x1_y",
        "x2_y",
        "y1_y",
        "y2_y",
    ]
)
# print(result)


clustering = DBSCAN(eps=0.5, min_samples=2).fit(result)
# print(clustering.labels_)

only_id["labels"] = clustering.labels_
clusters = [None] * len(set(clustering.labels_))
for x in range(len(set(clustering.labels_))):
    clusters[x] = []


for i in range(len(only_id)):
    clusters[only_id["labels"].values[i]].append(
        [only_id["_id_x"].values[i], only_id["_id_y"].values[i]]
    )

# PRINT CLUSTERS
# for i in range(len(clusters)):
#     print(i)
#     pprint(clusters[i])
#     print("--------------------------------")

# SCORING

scoringDict = {}

for i in range(len(dataOrg)):
    scoringDict[i + 1] = {}
    for j in range(len(clusters)):
        scoringDict[i + 1][j] = 0


# i = clusterNum, tpl = [box_a, box_b]
for i in range(len(clusters)):
    cl = clusters[i]
    for j in range(len(cl)):
        tpl = cl[j]
        box_a = cl[j][0]
        box_b = cl[j][1]

        scoringDict[box_a][i] += 1
        scoringDict[box_b][i] += 1

        # scoringDict[i][tpl[0]] += 1
        # scoringDict[i][tpl[1]] += 1


for boxId in scoringDict:
    sortedArr = sorted(
        scoringDict[boxId].items(), key=lambda item: item[1], reverse=True
    )
    scoringDict[boxId] = sortedArr


# pprint(scoringDict)


# /SCORING


cluster_ids = []
for i in range(len(clusters)):
    idlst = []
    for tup in clusters[i]:
        if tup[0] not in idlst:
            idlst.append(tup[0])
        if tup[1] not in idlst:
            idlst.append(tup[1])
    cluster_ids.append(idlst)

cluster_coords = []
for clst in cluster_ids:
    coord = []
    coordleftup = []
    coordrightup = []
    coordleftdown = []
    coordrightdown = []
    maxleftupx = 100000
    maxleftupy = 100000
    maxrightupx = 0
    maxrightupy = 100000
    maxleftdownx = 100000
    maxleftdowny = 0
    maxrightdownx = 0
    maxrightdowny = 0
    if clst == []:
        continue
    for a in clst:
        a = a - 1
        if corners["leftUpCorner"].iloc[a][0] <= maxleftupx:
            maxleftupx = corners["leftUpCorner"].iloc[a][0]
        if corners["leftUpCorner"].iloc[a][1] <= maxleftupy:
            maxleftupy = corners["leftUpCorner"].iloc[a][1]
        if corners["rightUpCorner"].iloc[a][0] >= maxrightupx:
            maxrightupx = corners["rightUpCorner"].iloc[a][0]
        if corners["rightUpCorner"].iloc[a][1] <= maxrightupy:
            maxrightupy = corners["rightUpCorner"].iloc[a][1]
        if corners["leftBottomCorner"].iloc[a][0] <= maxleftdownx:
            maxleftdownx = corners["leftBottomCorner"].iloc[a][0]
        if corners["leftBottomCorner"].iloc[a][1] >= maxleftdowny:
            maxleftdowny = corners["leftBottomCorner"].iloc[a][1]
        if corners["rightBottomCorner"].iloc[a][0] >= maxrightdownx:
            maxrightdownx = corners["rightBottomCorner"].iloc[a][0]
        if corners["rightBottomCorner"].iloc[a][1] >= maxrightdowny:
            maxrightdowny = corners["rightBottomCorner"].iloc[a][1]
    coordleftup.append(maxleftupx)
    coordleftup.append(maxleftupy)
    coordrightup.append(maxrightupx)
    coordrightup.append(maxrightupy)
    coordleftdown.append(maxleftdownx)
    coordleftdown.append(maxleftdowny)
    coordrightdown.append(maxrightdownx)
    coordrightdown.append(maxrightdowny)
    coord.append(coordleftup)
    coord.append(coordrightup)
    coord.append(coordleftdown)
    coord.append(coordrightdown)
    cluster_coords.append(coord)


def doOverlap(l1x, l1y, r1x, r1y, l2x, l2y, r2x, r2y):

    # To check if either rectangle is actually a line
    # For example  :  l1 ={-1,0}  r1={1,1}  l2={0,-1}  r2={0,1}

    if l1x == r1x or l1y == r2y or l2x == r2x or l2y == r2y:
        # the line cannot have positive overlap
        return False

    # If one rectangle is on left side of other
    if l1x >= r2x or l2x >= r1x:
        return False

    # If one rectangle is above other
    if l1y >= r2y or l2y >= r1y:
        return False

    return True


def isInside(l1x, l1y, r1x, r1y, l2x, l2y, r2x, r2y):
    if (
        (l2x <= l1x and l1x <= r2x)
        and (r1x >= l2x and r1x <= r2x)
        and (l2y <= l1y and l1y <= r2y)
        and (r1y >= l2y and r1y <= r2y)
    ):
        return True
    return False


# print(len(cluster_ids))

real_clusters = []
real_cluster_ids = []
clstid = 0
notclst = 0
realcl = 0
for clst in cluster_ids:
    if clst == []:
        continue
    l2x = cluster_coords[clstid][0][0]
    l2y = cluster_coords[clstid][0][1]
    r2x = cluster_coords[clstid][3][0]
    r2y = cluster_coords[clstid][3][1]
    isclst = False
    for a in range(len(corners)):
        a = a + 1
        if a in clst:
            if a == len(corners):
                isclst = True
            continue
        l1x = corners["leftUpCorner"].iloc[a - 1][0]
        l1y = corners["leftUpCorner"].iloc[a - 1][1]
        r1x = corners["rightBottomCorner"].iloc[a - 1][0]
        r1y = corners["rightBottomCorner"].iloc[a - 1][1]
        if doOverlap(l1x, l1y, r1x, r1y, l2x, l2y, r2x, r2y) or isInside(
            l1x, l1y, r1x, r1y, l2x, l2y, r2x, r2y
        ):
            notclst = notclst + 1
            break
        if a == len(corners):
            isclst = True
    if isclst == True:
        realcl = realcl + 1
        real_clusters.append(clst)
        real_cluster_ids.append(clstid)
    clstid = clstid + 1

for clst in real_clusters:
    clst.sort()
    # print(clst)
b_set = set(tuple(x) for x in real_clusters)
clusters_1 = [list(x) for x in b_set]
final_clusters = []
new_list = []
for clst in clusters_1:
    if len(clst) != len(corners):
        new_list.append(clst)
print(new_list)

# gets 2 object ids (_id_x and _id_y). check from sameRow['same_row'] value
# if the value is 1 return true else return false
def isSameRow(a, b):
    print("------------------")
    print(a,b)
    if (a < b):
        x = sameRowColumn.loc[sameRowColumn["_id_x"] == a]
        x = x.loc[x["_id_y"] == b]
        # print(x["same_row"].values[0])
        if x["same_row"].values[0] == "1":
            return True
        else:
            return False
    else:
        x = sameRowColumn.loc[sameRowColumn["_id_x"] == b]
        x = x.loc[x["_id_y"] == a]
        # print(x["same_row"].values[0])
        if x["same_row"].values[0] == "1":
            return True
        else:
            return False


# clst ayni rowda ise tum elemanlari, add to final cluster
print(new_list)
new_list_index = 0
for clst in new_list:
    sameRow = True
    for i in range(len(clst)):
        for j in range(i + 1, len(clst)):
            if isSameRow(clst[i], clst[j]):
                continue
            else:
                sameRow = False
                break
        if sameRow == False:
            break
    new_list_index += 1
    if sameRow == True:
        final_clusters.append(clst)
    if (new_list_index == len(new_list)):
        break
print(final_clusters)
quit()
# clst in kendisi disinda eleman sayisi ayni ise, o clst ile joinle; use center angle, ayni rowda olan ikilileri bir clst a at; add to final

for a in range(len(new_list)):
    huri = []
    sameLength = False
    for b in range(a + 1, len(new_list)):
        if len(new_list[a]) == len(new_list[b]):
            sameLength = True
        else:
            continue
    if sameLength == True:
        clsta = new_list[a]
        clstb = new_list[b]
        for i in clsta:
            for j in clstb:
                if isSameRow(i, j):
                    huri.append(i)
                    huri.append(j)
    final_clusters.append(huri)

# bu clst lar icinde olmayan objelerin id lerini bul, ayni rowda olanlari bir cluster a at.

item = []
huri2 = []
for i in range(len(corners)):
    item.append(i + 1)
unknown = []
for ct in new_list:
    for i in ct:
        huri2.append(i)

huri2 = set(huri2)

for element in item:
    if element not in huri2:
        unknown.append(element)

for i in range(len(unknown)):
    for j in range(i + 1, len(unknown)):
        new_cluster = []
        if isSameRow(i, j):
            new_cluster.append(i)
            new_cluster.append(j)
        final_clusters.append(new_cluster)


# final clst arrayindeki y degerlerini decremental orderda listele bu sana hiyerarsiyi vercek amk

print(final_clusters)
# pprint(real_clusters)
# print("=====================")
# print(real_cluster_ids)

# for clst in cluster_coords:
#     print(clst)

# print(cluster_coords)  # cluster bounding boxes coordinates

quit()
