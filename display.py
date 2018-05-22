"""
display.py

plotting the tracks in 3D
looking for the ways to unroll the helix

by Tianyi Miao
"""

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

print("finish importing; start running the script")


def plot_track_3d(df, n_tracks=10, cutoff=3):
    """
    :param df: Pandas DataFrame containing hit coordinate information
    must have the following columns: particle_id, [tx, ty, tz] or [x, y, z]

    :param n_tracks: the number of particles/tracks to display

    :param cutoff: particles with less than or equal to cutoff hits will be ignored.

    """

    if all("t"+s in df.columns for s in ("x", "y", "z")):
        xyz_cols = ["tx", "ty", "tz"]
    elif all(s in df.columns for s in ("x", "y", "z")):
        xyz_cols = ["x", "y", "z"]
    else:
        raise ValueError("input DataFrame does not contain valid coordinate columns (tx, ty, tz) or (x, y, z)")

    particle_list = np.unique(df.particle_id)
    particle_list = np.delete(particle_list, np.where(particle_list == 0))
    particle_list = np.random.choice(particle_list, size=n_tracks, replace=False)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for p_id in particle_list:
        coordinates = df.loc[df.particle_id == p_id, xyz_cols]

        if coordinates.shape[0] <= cutoff:
            # skip the particles with less than cutoff hits
            continue

        idx = np.argsort(coordinates[xyz_cols[-1]])  # sort by z value
        coordinates = coordinates[xyz_cols].values[idx]

        print("Particle ID: ", str(int(p_id)))
        print(coordinates)

        ax.plot(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], '.-')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_title("Particle ID: " + str(int(p_id)))
    plt.show()


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
    truth, = load_event(TRAIN_DIR + get_event_name(event_id), [TRUTH])
    plot_track_3d(truth, n_tracks=20, cutoff=3)
