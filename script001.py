"""
script001.py

basic exploratory data analysis

Author: Tianyi Miao
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


def add_features(hits_df):
    hits_df["r"] = np.sqrt(hits_df["x"]**2 + hits_df["y"]**2 + hits_df["z"]**2)
    hits_df["x*y"] = hits_df["x"] * hits_df["y"]
    hits_df["x*z"] = hits_df["x"] * hits_df["z"]
    hits_df["y*z"] = hits_df["y"] * hits_df["z"]
    # normalized
    hits_df["x/r"] = hits_df["x"] / hits_df["r"]
    hits_df["y/r"] = hits_df["y"] / hits_df["r"]
    hits_df["z/r"] = hits_df["z"] / hits_df["r"]


print("Start executing script001.py")

# train_1 ranges from 1000 to 2819 inclusive
test_events = np.random.choice(range(1000, 2820), size=1, replace=False)

cols = ["x", "y", "z"]

for event_id in test_events:
    hits, cells, particles, truth = load_event(TRAIN_DIR + get_event_name(event_id),
                                               [HITS, CELLS, PARTICLES, TRUTH])
    # dbscan_1 = cluster.DBSCAN(eps=0.5, min_samples=1, metric='euclidean', algorithm='auto', leaf_size=30, p=None, n_jobs=1)
    print("hits:\t\t", hits.columns.tolist())
    print("particles:\t\t", particles.columns.tolist())
    print("cells:\t\t", cells.columns.tolist())
    print("truth:\t\t", truth.columns.tolist())

    for ind in np.random.choice(range(particles.shape[0]), size=20, replace=False):
        temp_particle_id = particles.loc[ind, "particle_id"]
        print("temporary particle id:", temp_particle_id)
        print("number of hits:\t\t\t\t\t\t", particles.loc[ind, "nhits"], sep="")
        print("number of corresponding hit_ids:", end="\t")
        temp_hits = truth.loc[truth.particle_id == temp_particle_id, "hit_id"]
        print(len(temp_hits))
    """
    pred = pd.DataFrame({
        "hit_id": hits.hit_id,
        "track_id": dbscan_1.fit_predict(hits[cols])
    })
    """
    if False:
        # test scorer
        shuffled = shuffle_hits(truth, 0.05)  # 5% probability to reassign a hit
        print(shuffled)
        print(score_event(truth=truth, submission=shuffled))



