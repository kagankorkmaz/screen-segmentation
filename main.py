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

## Reading data
dataPath = "./data/kodi-base.csv"
dataOrg = pd.read_csv(dataPath)
data = dataOrg.copy()

## Calculating Corners, edges and cetnre values
############
data["x"] = data.apply(lambda row: (row.x1 + row.x2) / 2, axis=1)
data["y"] = data.apply(lambda row: (row.y1 + row.y2) / 2, axis=1) 

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
######################
data2 = data.copy()
data["key"] = 1
data2["key"] = 1

## Joining objects and creatign relations like angles, isSameRow, isSameColumn
############
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

## New dataframe for checking isSameRow or isSameColumn
################
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
###############

## New dataframe to getting object ids after clustering
#####################
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
###################
### calculating angles and distances for every corner, column and center
##################
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

## Eliminating unnecessary features before DBSCAN
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
#######################
#### Applying clustering
clustering = DBSCAN(eps=0.5, min_samples=2).fit(result)

only_id["labels"] = clustering.labels_
clusters = [None] * len(set(clustering.labels_))
### getting which objects are in which clusters after DBSCAN
for x in range(len(set(clustering.labels_))):
    clusters[x] = []

for i in range(len(only_id)):
    clusters[only_id["labels"].values[i]].append(
        [only_id["_id_x"].values[i], only_id["_id_y"].values[i]]
    )
############
# SCORING
## Scoring will be applied to decide which box in which cluster if needed(in the situations if heuristics can not decide for an object in which cluster)

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


cluster_ids = []
for i in range(len(clusters)):
    idlst = []
    for tup in clusters[i]:
        if tup[0] not in idlst:
            idlst.append(tup[0])
        if tup[1] not in idlst:
            idlst.append(tup[1])
    cluster_ids.append(idlst)

##################### Get bounding boxes for clusters
cluster_coords = [] # list of lists, for each cluster inner lists have [leftUpCorner, rightUpCorner, leftBottomCorner, rightBottomCorner] coordinates in order
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

##################### In order to check whether two bounding box are colliding or not
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

################ To check if one bounding box is inside of other
def isInside(l1x, l1y, r1x, r1y, l2x, l2y, r2x, r2y):
    if (
        (l2x <= l1x and l1x <= r2x)
        and (r1x >= l2x and r1x <= r2x)
        and (l2y <= l1y and l1y <= r2y)
        and (r1y >= l2y and r1y <= r2y)
    ):
        return True
    return False


############## Below we look at the clusters and their bounding boxes, if there is an object which is not an element of the cluster,
############## but inside/overlapping the bounding box of the cluster, we eliminate that cluster
real_clusters = []
real_cluster_ids = []
clstid = 0
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
            break
        if a == len(corners):
            isclst = True
    if isclst == True:
        real_clusters.append(clst)
        real_cluster_ids.append(clstid)
    clstid = clstid + 1

## real_clusters is the list that we have after the elimination

# We sort the clusters based on the element ids, in an incremental order, in place sort
for clst in real_clusters:
    clst.sort()
# After observing duplicate clusters, we eliminate them and make the rest unique
b_set = set(tuple(x) for x in real_clusters)
clusters_1 = [list(x) for x in b_set]
final_clusters = []
new_list = []
# As expected one of the clusters has the all objects, we eliminated that cluster below
for clst in clusters_1:
    if len(clst) != len(corners):
        new_list.append(clst)

# new_list is the final list of cluster before heuristics come take the show


# Gets 2 object ids (_id_x and _id_y) .it checks sameRow['same_row'] value for a and b ids.
# If the value is 1 return true else return false
def isSameRow(a, b):
    if (a < b):
        x = sameRowColumn.loc[sameRowColumn["_id_x"] == a]
        x = x.loc[x["_id_y"] == b]
        if x["same_row"].values[0] == "1":
            return True
        else:
            return False
    else:
        x = sameRowColumn.loc[sameRowColumn["_id_x"] == b]
        x = x.loc[x["_id_y"] == a]
        if x["same_row"].values[0] == "1":
            return True
        else:
            return False

############ 1ST HEURISTIC
# Check the clusters. 
# If all the elements of the cluster are in the samerow
# Take take cluster as a finalized cluster and add it to final_clusters list

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
print("\nAfter the first heuristic we get following finalized clusters: ")
print(final_clusters)
print("****************************")

############ 2ND HEURISTIC
# Take each cluster, compare them by their lengths
# If two clusters have same number of objects: Assumption, two clusters are represents two columns standing next to each other.
# Compare each elements of these clusters and if two objects are in the same row, add them to another list
# After filling the list with desired elements, add this list into final_clusters

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
    if huri == []:
        continue
    huri.sort()
    final_clusters.append(huri)

print("\nAfter the second heuristic we get the following finalized clusters: ")
print(final_clusters)
print("****************************")

############ 3RD HEURISTIC
# We observed that there are some objects missing in the clusters
# We assume that if those missing elements are in the same row, they make a new cluster, so we add them to final_clusters
# If those objects are not in the same row, we assume that they are seperate cluster of their own.

item = []
huri2 = []
for i in range(len(corners)):
    item.append(i + 1) # object ids are starting from 1
unknown = []
# we get all the object ids into huri2 list
for ct in new_list:
    for i in ct:
        huri2.append(i)
# delete the duplicates in huri2
huri2 = set(huri2)

# we find objects that were not present in the past clusters
for element in item:
    if element not in huri2:
        unknown.append(element)
# the list "unknown" has those uncaptured objects
for i in range(len(unknown)):
    for j in range(i + 1, len(unknown)):
        new_cluster = []
        if isSameRow(unknown[i], unknown[j]):
            new_cluster.append(unknown[i])
            new_cluster.append(unknown[j])
            new_cluster.sort()
            final_clusters.append(new_cluster)
        else:
            final_clusters.append([unknown[i]])
            final_clusters.append([unknown[j]])      

print("\nAfter the third heuristic we get the following finalized clusters: ")
print(final_clusters)
print("****************************")

############ 4TH HEURISTIC
# Now that we seperated every object in a different cluster, we need to find their hierarchical order
# Clusters' first elements give us the beginning of that region
# In the UI's like the one in the KODI example, the region in the above is the parent region of the below
# So we decided to use this knowledge, and sorted them in a decremental order according to clusters' first elements leftUpCorner y coordinate.
# In our case, from up to down, y coordinate increases (opposite of the general approach), from left to right x is increasing
# Thats why we assumed that the lowest y value will be the top cluster in the hierarchical order

index = 0
fclstid = []
# Here we take the cluster ids and the first element's leftUpCorner's y coordinate in a tuple and give it to fclstid list.
for clst in final_clusters:
    clsttup = []
    clsttup.append(index)
    clsttup.append(corners["leftUpCorner"].iloc[clst[0]-1][1])
    index += 1
    fclstid.append(clsttup)
# We sort the fclstid array based on the coordinates in an incremental order
fclstid.sort(key=lambda tup: tup[1])

# Below we printed out the hierarchical order for the final clusters.
print("\nHierarchical order for the finalized clusters after the 4th heuristic")
for i in range(len(fclstid)):
    print("****************************")
    print(str(i+1) + "th cluster: " + str(final_clusters[fclstid[i][0]]))
print("****************************")

quit()
