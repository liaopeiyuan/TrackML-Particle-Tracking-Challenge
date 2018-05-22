import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from trackml.dataset import load_event
from trackml.score import score_event

from arsenal import get_directories, get_event_name, StaticFeatureEngineer
from arsenal import HITS, CELLS, PARTICLES, TRUTH


TRAIN_DIR, TEST_DIR, DETECTORS_DIR, SAMPLE_SUBMISSION_DIR, TRAIN_EVENT_ID_LIST, TEST_EVENT_ID_LIST = get_directories(
    "E:/TrackMLData/"
)

n_event = 40  # important parameter
n_train = 1  # important parameter
event_id_list = np.random.choice(TRAIN_EVENT_ID_LIST, size=n_event, replace=False)
train_id_list = event_id_list[:n_train]  # training set
val_id_list = event_id_list[n_train:]  # validation set

for event_id in train_id_list:
    print('='*120)
    particles, truth = load_event(TRAIN_DIR + get_event_name(event_id), [PARTICLES, TRUTH])
    # merge truth and particle
    truth = truth.merge(particles, how="left", on="particle_id", copy=False)

    # drop noisy hits
    noisy_indices = truth[truth.particle_id == 0].index
    truth.drop(noisy_indices, axis=0, inplace=True)  # drop noisy hits
    
    particle_id_list = np.unique(truth.particle_id)
    
    selected_particle_id_list = np.random.choice(particle_id_list, size=1, replace=False)
    
    for p_id in selected_particle_id_list:
        hits_from_particle = truth.loc[truth.particle_id == p_id, ["tx", "ty", "tz", "hit_id"]]
        print(hits_from_particle)

