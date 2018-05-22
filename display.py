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


def plot_track_3d(df, transformer_list, n_tracks=10, cutoff=3, verbose=False):
    """
    :param df: Pandas DataFrame containing hit coordinate information
    must have the following columns: particle_id, [tx, ty, tz] or [x, y, z]

    :param n_tracks: the number of particles/tracks to display

    :param cutoff: particles with less than or equal to cutoff hits will be ignored.

    :param verbose: if set to True, print the particle id and the coordinate matrix in the console.
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
    for i, transformer in enumerate(transformer_list):
        ax = fig.add_subplot(1, len(transformer_list), i + 1, projection='3d')

        df_new = transformer(df)

        for p_id in particle_list:
            z = df.loc[df.particle_id == p_id, xyz_cols[-1]]  # original z coordinates

            if z.shape[0] <= cutoff:
                # skip the particles with less than cutoff hits
                continue

            idx = np.argsort(z)  # sort by z value
            coordinates = df_new.loc[df.particle_id == p_id, xyz_cols].values[idx]

            if verbose:
                print("Particle ID: ", str(int(p_id)))
                print(coordinates)
                print()

            ax.plot(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], '.-')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        # ax.set_title("Particle ID: " + str(int(p_id)))
    plt.show()


def test_dbscan(eps_list, hit_id, data, truth, scaling):
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


def transform_dummy(df):
    return df[["tx", "ty", "tz"]].copy()


def transform_1(df):
    new_df = df[["tx", "ty", "tz"]].copy()
    x = df["tx"]
    y = df["ty"]
    z = df["tz"]

    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)

    new_df["tx"] /= d
    new_df["ty"] /= d
    new_df["tz"] /= r
    return new_df

"""
def transform_1(df):
    x = df["tx"]
    y = df["ty"]
    z = df["tz"]

    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    # r = np.sqrt(df["tx"] ** 2 + df["ty"] ** 2)

    df["tx"] /= d
    df["ty"] /= d
    df["tz"] /= r

    # tz1 = np.sign(df["tz"]) * np.abs(df["tz"])**(1/2)
    # tz1 = df["tz"]
    # df["tx"] = df["tx"] / r  # * df["tz"]
    # df["ty"] = df["ty"] / r  # * df["tz"]

    # df["tz"] /= 4000
    # r2 = 10 ** -(np.sqrt(df["tx"] ** 2 + df["ty"] ** 2)) + 2

    # print(pd.Series(r2).describe())
    # df["tx"] = df["tx"] * r2
    # df["ty"] = df["ty"] * r2
    # df["tz"] = np.sign(df["tz"]) * np.sqrt(np.abs(df["tz"]))
    # ss = StandardScaler()
    # df[["tx", "ty", "tz"]] = ss.fit_transform(df[["tx", "ty", "tz"]])
    return None
"""

# TODO: important parameter
flag_plot = True

TRAIN_DIR, TEST_DIR, DETECTORS_DIR, SAMPLE_SUBMISSION_DIR, TRAIN_EVENT_ID_LIST, TEST_EVENT_ID_LIST = \
    get_directories("E:/TrackMLData/") if flag_plot else get_directories()

n_event = 40  # TODO:important parameter
n_train = 1 if flag_plot else 20  # TODO:important parameter
event_id_list = np.random.choice(TRAIN_EVENT_ID_LIST, size=n_event, replace=False)
train_id_list = event_id_list[:n_train]  # training set
val_id_list = event_id_list[n_train:]  # validation set


for event_id in train_id_list:
    print('='*120)
    truth, = load_event(TRAIN_DIR + get_event_name(event_id), [TRUTH])

    if flag_plot:
        plot_track_3d(truth, transformer_list=[transform_dummy, transform_1], n_tracks=100, cutoff=3)
    else:
        test_dbscan(
            # (0.001, 0.003, 0.008, 0.01, 0.02, 0.03, 0.07, 0.1, 0.3),
            (0.01, 0.03, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            hit_id=truth.hit_id, data=truth[["tx", "ty", "tz"]], truth=truth, scaling=False)

