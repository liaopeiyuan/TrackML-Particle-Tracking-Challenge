"""
script002.py

Use neural network for encoding

Author: Tianyi Miao

TrackML CheatSheet
hits:        ["hit_id", "x", "y", "z", "volume_id", "layer_id", "module_id"]
particles:   ["particle_id", "vx", "vy", "vz", "px", "py", "pz", "q", "nhits"]
cells:       ["hit_id", "ch0", "ch1", "value"]
truth:       ["hit_id", "particle_id", "tx", "ty", "tz", "tpx", "tpy", "tpz", "weight"]

nhit for a particle: minimum = 1, maximum = 28

"""

import os

import numpy as np
import pandas as pd

import itertools

from sklearn import cluster
from sklearn.preprocessing import StandardScaler
import hdbscan

from keras.layers import Input, Dense
from keras.models import Model

from trackml.dataset import load_event
from trackml.score import score_event

from arsenal import get_directories, get_event_name, StaticFeatureEngineer
from arsenal import HITS, CELLS, PARTICLES, TRUTH

# define important directories; change it if you store your data differently!
# type help(get_directories) for more information
# TRAIN_DIR, TEST_DIR, DETECTORS_DIR, SAMPLE_SUBMISSION_DIR, TRAIN_EVENT_ID_LIST, TEST_EVENT_ID_LIST = get_directories("E:/TrackMLData/")
TRAIN_DIR, TEST_DIR, DETECTORS_DIR, SAMPLE_SUBMISSION_DIR, TRAIN_EVENT_ID_LIST, TEST_EVENT_ID_LIST = get_directories()

# load feature engineering
def get_feature_engineer():
    sfe1 = StaticFeatureEngineer()
    sfe1.add_method("r", lambda df: np.sqrt(df.x**2 + df.y**2 + df.z**2))
    sfe1.add_method("rx", lambda df: np.sqrt(df.y**2 + df.z**2))
    sfe1.add_method("ry", lambda df: np.sqrt(df.x**2 + df.z**2))
    sfe1.add_method("rz", lambda df: np.sqrt(df.x**2 + df.y**2))
    # product and quotient
    # for v1, v2 in (("x", "y"), ("y", "z"), ("z", "x")):
        # sfe1.add_method("{}*{}".format(v1, v2), lambda df: df[v1] * df[v2])
        # sfe1.add_method("{}/{}".format(v1, v2), lambda df: df[v1] / df[v2])
    # normalized
    for v1, v2 in itertools.product(["x", "y", "z"], ["r", "rx", "ry", "rz"]):
        sfe1.add_method("{}/{}".format(v1, v2), lambda df: df[v1] / df[v2])
    # angles
    for v1, v2 in (("x", "y"), ("y", "z"), ("z", "x")):
        sfe1.add_method("angle-{}-{}".format(v1, v2), lambda df: (df[v2] < 0) * np.pi + np.arctan(df[v1]/df[v2]))
    sfe1.compile()
    return sfe1

sfe1 = get_feature_engineer()
print(sfe1.get_n_variables())
print(sfe1.get_variables())


n_event = 10
n_train = 5
event_id_list = np.random.choice(TRAIN_EVENT_ID_LIST, size=n_event, replace=False)
train_id_list = event_id_list[:n_train]  # training set
val_id_list = event_id_list[n_train:]  # validation set


# TODO: load keras model
n_features = 3 + sfe1.get_n_variables()
feature_cols = ["x", "y", "z"] + sfe1.get_variables()
# target_cols = ["vx", "vy", "vz", "px", "py", "pz", "q"]
target_cols = ["tx", "ty", "tz", "tpx", "tpy", "tpz"]


input_layer = Input(shape=(n_features,))
encoded = Dense(256, activation="relu")(input_layer)
encoded = Dense(128, activation="relu")(encoded)
encoded = Dense(96, activation="relu")(encoded)
encoded = Dense(64, activation="relu")(encoded)
encoded = Dense(32, activation="relu")(encoded)
encoded = Dense(16, activation="sigmoid")(encoded)

decoded = Dense(32, activation="relu")(encoded)
decoded = Dense(64, activation="relu")(decoded)
decoded = Dense(96, activation="relu")(decoded)
decoded = Dense(len(target_cols), activation="linear")(decoded)

encoder = Model(input_layer, encoded)

nn_predictor = Model(input_layer, decoded)
nn_predictor.compile(optimizer="adadelta", loss="mean_squared_error")


for event_id in train_id_list:
    # important observation:
    # Many particle_id in particles do not appear in truth; perhaps they are not detected at all
    # One particle_id in truth does not appear in particles: 0.

    hits, particles, truth = load_event(TRAIN_DIR + get_event_name(event_id), [HITS, PARTICLES, TRUTH])
    noisy_indices = truth[truth.particle_id == 0].index
    hits.drop(noisy_indices, axis=0, inplace=True)  # drop noisy hits
    truth.drop(noisy_indices, axis=0, inplace=True)  # drop noisy hits

    truth = truth.merge(particles, how="left", on="particle_id", copy=False)
    # print(truth.columns)
    # print(truth.describe())

    hits = sfe1.transform(hits, copy=False)

    nn_predictor.fit(
        x=hits[feature_cols],
        y=truth[target_cols],
        batch_size=256, epochs=20, shuffle=True, validation_split=0.2)

# start evaluation
for event_id in val_id_list:
    hits, truth = load_event(TRAIN_DIR + get_event_name(event_id), [HITS, TRUTH])
    hits = sfe1.transform(hits, copy=False)
    X_new = encoder.predict(hits[feature_cols])

    # dbscan_1 = cluster.DBSCAN(eps=0.1, min_samples=1, algorithm='auto', n_jobs=-1)
    dbscan_2 = hdbscan.HDBSCAN(min_samples=2, min_cluster_size=7, cluster_selection_method='leaf',
                               prediction_data=False, metric='braycurtis', core_dist_n_jobs=-1)
    print("start predicting", end="")
    pred = pd.DataFrame({
        "hit_id": hits.hit_id,
        "track_id": dbscan_2.fit_predict(X_new)
    })

    print("\rfinal score:", end='\t')
    print(score_event(truth=truth, submission=pred))