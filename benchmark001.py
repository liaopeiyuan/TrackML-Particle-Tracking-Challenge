"""
benchmark001.py

Naive DBSCAN script, completely unsupervised learning

Author: Tianyi Miao

TrackML CheatSheet
hits:        ['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'module_id']
particles:   ['particle_id', 'vx', 'vy', 'vz', 'px', 'py', 'pz', 'q', 'nhits']
cells:       ['hit_id', 'ch0', 'ch1', 'value']
truth:       ['hit_id', 'particle_id', 'tx', 'ty', 'tz', 'tpx', 'tpy', 'tpz', 'weight']

nhit for a particle: minimum = 1, maximum = 28

"""

import numpy as np
import pandas as pd

from sklearn import cluster
from sklearn.preprocessing import StandardScaler

import hdbscan

# import xgboost as xgb
# import lightgbm as lgbm

from trackml.dataset import load_event
from trackml.score import score_event

from arsenal import get_directories, get_event_name
from arsenal import HITS, CELLS, PARTICLES, TRUTH

# important constants
# define important directories; change it if you store your data differently!
# TRAIN_DIR, TEST_DIR, DETECTORS_DIR, SAMPLE_SUBMISSION_DIR, TRAIN_EVENT_ID_LIST, TEST_EVENT_ID_LIST = get_directories("E:/TrackMLData/")
TRAIN_DIR, TEST_DIR, DETECTORS_DIR, SAMPLE_SUBMISSION_DIR, TRAIN_EVENT_ID_LIST, TEST_EVENT_ID_LIST = get_directories()


def add_features(hits_df, copy=True):
    if copy:
        hits_df = hits_df.copy()
    hits_df["r"] = np.sqrt(hits_df["x"]**2 + hits_df["y"]**2 + hits_df["z"]**2)
    hits_df["x*y"] = hits_df["x"] * hits_df["y"]
    hits_df["x*z"] = hits_df["x"] * hits_df["z"]
    hits_df["y*z"] = hits_df["y"] * hits_df["z"]
    # normalized
    hits_df["x/r"] = hits_df["x"] / hits_df["r"]
    hits_df["y/r"] = hits_df["y"] / hits_df["r"]
    hits_df["z/r"] = hits_df["z"] / hits_df["r"]
    return hits_df


def normalize_1(hits_df, copy=True):
    if copy:
        hits_df = hits_df.copy()
    hits_df["r"] = np.sqrt(hits_df["x"] ** 2 + hits_df["y"] ** 2 + hits_df["z"] ** 2)
    hits_df["x"] /= hits_df["r"]
    hits_df["y"] /= hits_df["r"]
    hits_df["z"] /= hits_df["r"]
    return hits_df


def normalize_2(hits_df, copy=True):
    if copy:
        hits_df = hits_df.copy()
    r = np.sqrt(hits_df["x"] ** 2 + hits_df["y"] ** 2 + hits_df["z"] ** 2)
    rz = np.sqrt(hits_df["x"] ** 2 + hits_df["y"] ** 2)
    hits_df["x"] /= r
    hits_df["y"] /= r
    hits_df["z"] /= rz
    return hits_df


def normalize_3(hits, rz_scale=1.0, copy=False):
    x = hits.x.values
    y = hits.y.values
    z = hits.z.values

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    hits['x2'] = x / r
    hits['y2'] = y / r
    r = np.sqrt(x ** 2 + y ** 2)
    hits['z2'] = z / r

    ss = StandardScaler()
    X = ss.fit_transform(hits[['x2', 'y2', 'z2']].values)
    X[:, 2] = X[:, 2] * rz_scale

    return X


print("Start executing benchmark001.py")

# train_1 ranges from 1000 to 2819 inclusive
event_id_list = np.random.choice(TRAIN_EVENT_ID_LIST, size=15, replace=False)

cols = ["x", "y", "z"]

sc1 = StandardScaler()
dbscan_1 = cluster.DBSCAN(eps=0.00715, min_samples=1, algorithm='auto', n_jobs=-1)
dbscan_2 = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=3, cluster_selection_method='leaf',
                           prediction_data=False, metric='braycurtis', core_dist_n_jobs=-1)
for event_id in event_id_list:
    # hits, cells, particles, truth = load_event(TRAIN_DIR + get_event_name(event_id), [HITS, CELLS, PARTICLES, TRUTH])
    hits, truth = load_event(TRAIN_DIR + get_event_name(event_id), [HITS, TRUTH])
    # dbscan_1 = cluster.DBSCAN(eps=0.01, min_samples=3, metric='euclidean', algorithm='auto', leaf_size=30, p=None, n_jobs=1)
    hits2 = normalize_2(hits, copy=False)
    # print(hits[cols].describe())
    print("predicting...")

    for min_cluster_size in range(2, 12):
        dbscan_2 = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=min_cluster_size, cluster_selection_method='leaf',
                                   prediction_data=False, metric='braycurtis', core_dist_n_jobs=-1)
        pred = pd.DataFrame({
            "hit_id": hits.hit_id,
            "track_id": dbscan_2.fit_predict(StandardScaler().fit_transform(hits2[cols]))
        })
        # print(pred)
        # this should give an average score of 0.2
        print("c = {}; final score:\t".format(min_cluster_size), end="")
        print(score_event(truth=truth, submission=pred))

    dbscan_1 = cluster.DBSCAN(eps=0.00715, min_samples=1, algorithm='auto', n_jobs=-1)
    pred = pd.DataFrame({
        "hit_id": hits.hit_id,
        "track_id": dbscan_1.fit_predict(StandardScaler().fit_transform(hits2[cols]))
    })
    print("dbscan; final score:\t", end="")
    print(score_event(truth=truth, submission=pred))

