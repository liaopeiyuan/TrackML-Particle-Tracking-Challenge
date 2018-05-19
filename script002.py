"""
script002.py

Author: Tianyi Miao

TrackML CheatSheet
hits:        ['hit_id', 'x', 'y', 'z', 'volume_id', 'layer_id', 'module_id']
particles:   ['particle_id', 'vx', 'vy', 'vz', 'px', 'py', 'pz', 'q', 'nhits']
cells:       ['hit_id', 'ch0', 'ch1', 'value']
truth:       ['hit_id', 'particle_id', 'tx', 'ty', 'tz', 'tpx', 'tpy', 'tpz', 'weight']

nhit for a particle: minimum = 1, maximum = 28

"""

import os

import numpy as np
import pandas as pd

import itertools

from sklearn import cluster
from sklearn.preprocessing import StandardScaler

from keras.layers import Input, Dense
from keras.models import Model

from trackml.dataset import load_event
from trackml.score import score_event

from arsenal import get_event_name, StaticFeatureEngineer

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

# there are 8850 events in the training dataset; some ids from 1000 to 9999 are skipped
TRAIN_EVENT_ID_LIST = sorted(set(int(x[x.index("0"):x.index("-")]) for x in os.listdir(TRAIN_DIR)))
TEST_EVENT_ID_LIST = sorted(set(int(x[x.index("0"):x.index("-")]) for x in os.listdir(TEST_DIR)))

# load feature engineering
sfe1 = StaticFeatureEngineer()
sfe1.add_method("r", lambda df: np.sqrt(df.x**2 + df.y**2 + df.z**2))
sfe1.add_method("rx", lambda df: np.sqrt(df.y**2 + df.z**2))
sfe1.add_method("ry", lambda df: np.sqrt(df.x**2 + df.z**2))
sfe1.add_method("rz", lambda df: np.sqrt(df.x**2 + df.y**2))
# product and quotient
for v1, v2 in itertools.combinations(["x", "y", "z"], r=2):
    sfe1.add_method("{}*{}".format(v1, v2), lambda df: df[v1] * df[v2])
    # sfe1.add_method("{}/{}".format(v1, v2), lambda df: df[v1] / df[v2])
# normalized
for v1, v2 in itertools.product(["x", "y", "z"], ["r", "rx", "ry", "rz"]):
    sfe1.add_method("{}/{}".format(v1, v2), lambda df: df[v1] / df[v2])
sfe1.compile()

print(sfe1.get_n_variables())
print(sfe1.get_variables())


n_event = 40
n_train = 20
event_id_list = np.random.choice(TRAIN_EVENT_ID_LIST, size=n_event, replace=False)
train_id_list = event_id_list[:n_train]  # training set
val_id_list = event_id_list[n_train:]  # validation set


# TODO: load keras model
input_dim = 3 + sfe1.get_n_variables()
input_cols = ["x", "y", "z"] + sfe1.get_variables()



for event_id in event_id_list:
    hits, truth = load_event(TRAIN_DIR + get_event_name(event_id), [HITS, TRUTH])
    hits = sfe1.transform(hits, copy=False)
    for col in hits.columns:
        print(hits[col].describe())
        print("\n\n")
    break
