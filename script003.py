"""
script003.py

continue with hidden space strategy
important: error analysis by parts

by Tianyi Miao
"""

import numpy as np
import pandas as pd

import itertools

from sklearn import cluster
from sklearn.preprocessing import StandardScaler, LabelEncoder
import hdbscan

from trackml.dataset import load_event
from trackml.score import score_event

from arsenal import get_directories, get_event_name, StaticFeatureEngineer
from arsenal import HITS, CELLS, PARTICLES, TRUTH


# define important directories; change it if you store your data differently!
# type help(get_directories) for more information
# TRAIN_DIR, TEST_DIR, DETECTORS_DIR, SAMPLE_SUBMISSION_DIR, TRAIN_EVENT_ID_LIST, TEST_EVENT_ID_LIST = get_directories("E:/TrackMLData/")
TRAIN_DIR, TEST_DIR, DETECTORS_DIR, SAMPLE_SUBMISSION_DIR, TRAIN_EVENT_ID_LIST, TEST_EVENT_ID_LIST = get_directories()


n_event = 10
n_train = 5
event_id_list = np.random.choice(TRAIN_EVENT_ID_LIST, size=n_event, replace=False)
train_id_list = event_id_list[:n_train]  # training set
val_id_list = event_id_list[n_train:]  # validation set


# c is short for generic x/y/z
c_cols = ["x", "y", "z"]
vlm_cols = ["volume_id", "layer_id", "module_id"]
vc_cols = ["vx", "vy", "vz"]
pc_cols = ["px", "py", "pz"]
tc_cols = ["tx", "ty", "tz"]
tpc_cols = ["tpx", "tpy", "tpz"]

for event_id in train_id_list:
    # important observation:
    # Many particle_id in particles do not appear in truth; perhaps they are not detected at all
    # One particle_id in truth does not appear in particles: 0.

    hits, particles, truth = load_event(TRAIN_DIR + get_event_name(event_id), [HITS, PARTICLES, TRUTH])
    noisy_indices = truth[truth.particle_id == 0].index
    hits.drop(noisy_indices, axis=0, inplace=True)  # drop noisy hits
    truth.drop(noisy_indices, axis=0, inplace=True)  # drop noisy hits

    truth = truth.merge(particles, how="left", on="particle_id", copy=False)
    # TODO: drop useless columns==================
    truth.drop(tc_cols + tpc_cols, axis=1, inplace=True)
    hits.drop(c_cols + vlm_cols, axis=1, inplace=True)
    # TODO: ======================================
    X_new = truth[pc_cols]

    # categorical encoding: use LabelEncoder
    """
    print("final score:    ", end="")
    pred = pd.DataFrame({
        "hit_id": hits.hit_id,
        "track_id": LabelEncoder().fit_transform([str(row) for row in X_new.values])
        # "track_id": truth.particle_id
    })
    print(score_event(truth=truth, submission=pred))
    """

    use_hdbscan = True
    print("start predicting...")
    if use_hdbscan:  # use DBSCAN instead
        for min_cluster_size in range(2, 10):
            dbscan_2 = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=min_cluster_size, cluster_selection_method='leaf',
                                       prediction_data=False, metric='braycurtis', core_dist_n_jobs=-1)

            pred = pd.DataFrame({
                "hit_id": hits.hit_id,
                "track_id": dbscan_2.fit_predict(X_new)
            })
            print("n={}, final score:".format(min_cluster_size), end="    ")
            print(score_event(truth=truth, submission=pred))
    else:
        dbscan_1 = cluster.DBSCAN(eps=0.00715, min_samples=1, algorithm='auto', n_jobs=-1)
        pred = pd.DataFrame({
            "hit_id": hits.hit_id,
            "track_id": dbscan_1.fit_predict(X_new)
        })
        print("dbscan, final score:", end="    ")
        print(score_event(truth=truth, submission=pred))

# print("Error Analysis 2: cluster tracks from ")