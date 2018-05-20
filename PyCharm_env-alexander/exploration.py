"""
script001.py

basic exploratory data analysis

Author: Tianyi Miao

TrackML CheatSheet
hits:        ['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'module_id']
particles:   ['particle_id', 'vx', 'vy', 'vz', 'px', 'py', 'pz', 'q', 'nhits']
cells:       ['hit_id', 'ch0', 'ch1', 'value']
truth:       ['hit_id', 'particle_id', 'tx', 'ty', 'tz', 'tpx', 'tpy', 'tpz', 'weight']



"""
import os

import numpy as np
import pandas as pd

from sklearn import cluster

# import xgboost as xgb
# import lightgbm as lgbm

from trackml.dataset import load_event
from trackml.score import score_event
from trackml.randomize import shuffle_hits

# important constants
# define data name strings as constants; prevent spelling errors
HITS = "hits"
CELLS = "cells"
PARTICLES = "particles"
TRUTH = "truth"
# define important directories; change it if you store your data differently!
TRAIN_DIR = "E:/TrackMLData/train/"  # directory containing training dataset
TEST_DIR = "E:/TrackMLData/test/"  # directory containing test dataset
DETECTORS_DIR = "E:/TrackMLData/detectors.csv"  # csv file for detectors
SAMPLE_SUBMISSION_DIR = "E:/TrackMLData/sample_submission.csv"  # csv file for sample submission


def get_event_name(event_id):
    return "event" + str(event_id).zfill(9)


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


def normalize_features(hits_df, copy=True):
    if copy:
        hits_df = hits_df.copy()
    hits_df["r"] = np.sqrt(hits_df["x"] ** 2 + hits_df["y"] ** 2 + hits_df["z"] ** 2)
    hits_df["x"] /= hits_df["r"]
    hits_df["y"] /= hits_df["r"]
    hits_df["z"] /= hits_df["r"]
    return hits_df

print("Start executing script001.py")

# train_1 ranges from 1000 to 2819 inclusive
test_events = np.random.choice(range(1000, 2820), size=1, replace=False)

cols = ["x", "y", "z"]

for event_id in test_events:
    # hits, cells, particles, truth = load_event(TRAIN_DIR + get_event_name(event_id), [HITS, CELLS, PARTICLES, TRUTH])
    hits, truth = load_event(TRAIN_DIR + get_event_name(event_id), [HITS, TRUTH])

    dbscan_1 = cluster.DBSCAN(eps=0.1, min_samples=1, metric='euclidean', algorithm='auto', leaf_size=30, p=None, n_jobs=1)

    normalize_features(hits, copy=False)
    print("start predicting")

    pred = pd.DataFrame({
        "hit_id": hits.hit_id,
        "track_id": dbscan_1.fit_predict(hits[cols])
    })

    print(pred)
    print("final score:")
    print(score_event(truth=truth, submission=pred))

