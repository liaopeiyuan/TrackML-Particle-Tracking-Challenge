"""
script004.py

use decision trees and random forests to explore the relationship between tc_cols, tpc_cols, and pc_cols
continue from script003.py
start a new file to prevent complication

by Tianyi Miao
"""

import numpy as np
import pandas as pd

import itertools

from sklearn.cluster import DBSCAN, Birch, AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from trackml.dataset import load_event
from trackml.score import score_event

from arsenal import get_directories, get_event_name, StaticFeatureEngineer
from arsenal import HITS, CELLS, PARTICLES, TRUTH


TRAIN_DIR, TEST_DIR, DETECTORS_DIR, SAMPLE_SUBMISSION_DIR, TRAIN_EVENT_ID_LIST, TEST_EVENT_ID_LIST = get_directories()

# hits level
c_cols = ["x", "y", "z"]
vlm_cols = ["volume_id", "layer_id", "module_id"]
# truth level
tc_cols = ["tx", "ty", "tz"]
tpc_cols = ["tpx", "tpy", "tpz"]
# particles level
vc_cols = ["vx", "vy", "vz"]
pc_cols = ["px", "py", "pz"]


def test_dbscan(eps_list, hit_id, data, scaling):
    for eps in eps_list:
        dbscan_1 = DBSCAN(eps=eps, min_samples=1, algorithm='auto', n_jobs=-1)
        pred = pd.DataFrame({
            "hit_id": hit_id,
            "track_id": dbscan_1.fit_predict(
                StandardScaler().fit_transform(data) if scaling else data
            )
        })
        print("eps={}, score:  ".format(eps), end='\t')
        print(score_event(truth=truth, submission=pred))


def get_feature_engineer():
    sfe = StaticFeatureEngineer()
    sfe.add_method("r", lambda df: np.sqrt(df.tx ** 2 + df.ty ** 2 + df.tz ** 2))
    sfe.add_method("rx", lambda df: np.sqrt(df.ty ** 2 + df.tz ** 2))
    sfe.add_method("ry", lambda df: np.sqrt(df.tx ** 2 + df.tz ** 2))
    sfe.add_method("rz", lambda df: np.sqrt(df.tx ** 2 + df.ty ** 2))
    sfe.compile()
    return sfe

sfe1 = get_feature_engineer()

feature_cols = tc_cols + sfe1.get_variables()  # TODO: important parameter
target_cols = pc_cols

n_event = 30  # TODO: important parameter
n_train = 20  # TODO: important parameter
event_id_list = np.random.choice(TRAIN_EVENT_ID_LIST, size=n_event, replace=False)
train_id_list = event_id_list[:n_train]  # training set
val_id_list = event_id_list[n_train:]  # validation set

tree_delta = 10  # TODO: important parameter (the number of trees added at each training iteration)

rf_predictor = RandomForestRegressor(n_estimators=tree_delta, criterion="mse", max_depth=None, max_features=0.8,
                                     min_samples_split=20,
                                     n_jobs=-1, warm_start=True, verbose=0)
dt_predictor = DecisionTreeRegressor(criterion="mse", splitter="best", max_depth=None, min_samples_leaf=80)

avg_val_score = 0.0
avg_train_score = 0.0

predictor = rf_predictor  # TODO: important parameter

for event_id in train_id_list:
    print('='*120)
    particles, truth = load_event(TRAIN_DIR + get_event_name(event_id), [PARTICLES, TRUTH])
    truth = truth.merge(particles, how="left", on="particle_id", copy=False)
    n_particles = np.unique(truth.particle_id).size

    # drop noisy hits
    noisy_indices = truth[truth.particle_id == 0].index
    n_particles -= 1  # important: "0" (noisy hits) is dropped
    truth.drop(noisy_indices, axis=0, inplace=True)  # drop noisy hits

    # drop useless columns
    truth.drop(vc_cols + ["nhits"], axis=1, inplace=True)

    # TODO: important; add feature engineering
    sfe1.transform(truth, copy=False)

    try:
        val_score = predictor.score(X=truth[feature_cols], y=truth[target_cols])
        avg_val_score += val_score
        print("validation score: {}".format(val_score))
    except Exception as err:
        print(err)

    predictor.fit(X=truth[feature_cols], y=truth[target_cols])

    train_score = predictor.score(X=truth[feature_cols], y=truth[target_cols])
    print("training score: {}".format(train_score))
    avg_train_score += train_score

    if isinstance(predictor, RandomForestRegressor):
        if predictor.warm_start:
            predictor.n_estimators += tree_delta
        print("number of trees: {}".format(len(predictor.estimators_)))

avg_val_score /= len(train_id_list) - 1
avg_train_score /= len(train_id_list)
print("average training score: ", avg_train_score)
print("average validation score: ", avg_val_score)

for event_id in val_id_list:
    print('=' * 120)
    particles, truth = load_event(TRAIN_DIR + get_event_name(event_id), [PARTICLES, TRUTH])
    truth = truth.merge(particles, how="left", on="particle_id", copy=False)
    n_particles = np.unique(truth.particle_id).size

    # drop noisy hits
    noisy_indices = truth[truth.particle_id == 0].index
    n_particles -= 1  # important: "0" (noisy hits) is dropped
    truth.drop(noisy_indices, axis=0, inplace=True)  # drop noisy hits

    # drop useless columns
    truth.drop(vc_cols + ["nhits"], axis=1, inplace=True)

    X_new = predictor.predict(X=truth[feature_cols])
    print(pd.DataFrame(X_new).describe())
    print(truth[target_cols].describe())

    test_dbscan((0.001, 0.003, 0.008, 0.01, 0.02, 0.03, 0.1), truth.hit_id, X_new, scaling=False)

