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
# data["y"] = data.apply(lambda row: (row.y1 + row.y2) / 2, axis=1)  # (x1,y)
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

data = data.drop(columns=['x1', 'x2', 'y1', 'y2', 'inside', 'width', 'height'])

data2 = data.copy()
data["key"] = 1
data2["key"] = 1
result = pd.merge(data, data2, on="key").drop("key", 1)
result = result[result["_id_x"] < result["_id_y"]]

for a in range(9):
    first=""
    if a==0:
        first = 'leftUpCorner_x'
    elif a==1:
        first = 'rightUpCorner_x'
    elif a==2:
        first = 'leftBottomCorner_x'
    elif a==3:
        first = 'rightBottomCorner_x'
    elif a==4:
        first = 'topEdgeCenter_x'
    elif a==5:
        first = 'BottomEdgeCenter_x'
    elif a==6:
        first = 'leftEdgeCenter_x'
    elif a==7:
        first = 'rightEdgeCenter_x'
    else:
        first = 'center_x' 
    for b in range(9):
        column_index = a*9+b
        column_name = "diff_"+str(column_index)
        angle_name = "angle_"+str(column_index)
        
        second=""
        if b == 0:
            second = 'leftUpCorner_y'
        elif b==1:
            second = 'rightUpCorner_y'
        elif b==2:
            second = 'leftBottomCorner_y'
        elif b==3:
            second = 'rightBottomCorner_y'
        elif b==4:
            second = 'topEdgeCenter_y'
        elif b==5:
            second = 'BottomEdgeCenter_y'
        elif b==6:
            second = 'leftEdgeCenter_y'
        elif b==7:
            second = 'rightEdgeCenter_y'
        else:
            second = 'center_y'    
        
        # result[column_name] = result.apply(
        #     lambda row: math.sqrt(pow((row.loc(first)[0] - row.loc(second)[0]), 2) + pow((row.loc(first)[1] - row.loc(second)[1]), 2)),
        #     axis=1,
        # )
        ls = []
        ls_angle = []
        for c in range(len(result.index)):
            ls.append(math.sqrt(pow((result[first].iloc[c][0] - result[second].iloc[c][0]), 2) + pow((result[first].iloc[c][1] - result[second].iloc[c][1]), 2)))
            ls_angle.append(np.rad2deg(np.arctan2((result[first].iloc[c][1] - result[second].iloc[c][1]), (result[first].iloc[c][0] - result[second].iloc[c][0]))))
        result[column_name]=ls
        result[angle_name]= ls_angle
        result[column_name] = MinMaxScaler().fit_transform(
            np.array(result[column_name]).reshape(-1, 1)
            )
        result[angle_name] = MinMaxScaler().fit_transform(
            np.array(result[angle_name]).reshape(-1, 1)
            )
# print(result['leftUpCorner_x'].iloc[0][1])    
print(result)
quit()


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
    clusters[only_id["labels"].values[i]].append(
        [only_id["_id_x"].values[i], only_id["_id_y"].values[i]]
    )

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(only_id)

# for i in range(len(clusters)):
#     print(i)
#     pprint(clusters[i])
#     print("--------------------------------")

quit()