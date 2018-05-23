"""
display.py

plotting the tracks in 3D
looking for the ways to unroll the helix

transform_1 steadily gives a DBSCAN score around 0.2, when eps=0.008 and scaling=True

by Tianyi Miao
"""
from collections import Counter

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import DBSCAN, dbscan
from sklearn.preprocessing import StandardScaler

from trackml.dataset import load_event
from trackml.score import score_event

from arsenal import get_directories, get_event_name, StaticFeatureEngineer
from arsenal import HITS, CELLS, PARTICLES, TRUTH

print("finish importing; start running the script")


def plot_track_3d(df, transformer_list, clusterer_list=(), n_tracks=10, cutoff=3, verbose=False):
    """
    :param df: Pandas DataFrame containing hit coordinate information
    must have the following columns: particle_id, [tx, ty, tz] or [x, y, z]

    :param transformer_list: list of transformer functions that takes df as input, and returns a copy of df[xyz_cols]

    :param clusterer_list: a list of clusterer objects; each of them will be used to transform the new dataset
    example: [DBSCAN(eps=0.01), DBSCAN(eps=0.1)]

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
    i = 1

    p_masks = {p_id: df.particle_id == p_id for p_id in particle_list}
    # filter out particles with fewer hits than cutoff
    p_masks = {p_id: p_masks[p_id] for p_id in p_masks if sum(p_masks[p_id]) > cutoff}

    p_union_mask = np.sum([p_masks[p_id] for p_id in p_masks], axis=0).astype(bool)

    for transformer in transformer_list:  # iterate over transformers
        ax = fig.add_subplot(len(transformer_list), len(clusterer_list) + 1, i, projection='3d')
        # transform the dataset using transformer
        df_new = transformer(df)

        for p_id in p_masks:
            z = df.loc[p_masks[p_id], xyz_cols[-1]]  # original z coordinates
            idx = np.argsort(z)  # sort by z value
            coordinates = df_new.loc[p_masks[p_id], xyz_cols].values[idx]

            if verbose:
                print("Particle ID: ", str(int(p_id)))
                print(coordinates)
                print()

            ax.plot(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], '.-')

        ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
        # ax.set_title("Particle ID: " + str(int(p_id)))
        i += 1  # increment the subplot index

        for clusterer in clusterer_list:
            ax = fig.add_subplot(len(transformer_list), len(clusterer_list) + 1, i, projection='3d')
            cluster_id = clusterer.fit_predict(df_new)
            for c_id in np.unique(cluster_id[p_union_mask]):
                c_mask = (cluster_id == c_id) & p_union_mask
                z = df.loc[c_mask, xyz_cols[-1]]
                idx = np.argsort(z)  # sort by z value
                coordinates = df_new.loc[c_mask, xyz_cols].values[idx]
                ax.plot(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], '.-')

            """
            for p_id in particle_list:
                p_mask = df.particle_id == p_id  # boolean array denoting the occurrences of p_id
                z = df.loc[p_mask, xyz_cols[-1]]  # original z coordinates

                if z.shape[0] <= cutoff:
                    # skip the particles with less than cutoff hits
                    continue

                idx = np.argsort(z)  # sort by z value
                coordinates = df_new.loc[p_mask, xyz_cols].values[idx]

                ax.plot(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], '.-', c=cluster_id[p_mask][idx])
            """
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            i += 1
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


def helix_1(x, y, z):
    """
    helix unrolling that works since the beginning
    """
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    return x/d, y/d, z/r


def helix_2(x, y, z, theta=20):
    """
    rotate the helix by an angle of 20
    """
    phi = theta / 180 * np.pi * z * 0.0005  # normalize z into range [-1, 1]
    hx = x * np.cos(-phi) - y * np.sin(-phi)
    hy = x * np.sin(-phi) + y * np.cos(-phi)
    return hx, hy, z


def transform_dummy(df, scaling=False):
    xyz_cols = ["tx", "ty", "tz"]
    new_df = df[xyz_cols].copy()
    if scaling:
        new_df[xyz_cols] = StandardScaler().fit_transform(new_df[xyz_cols])
    return new_df


def transform_1(df, scaling=False):
    xyz_cols = ["tx", "ty", "tz"]
    new_df = df[xyz_cols].copy()
    new_df[xyz_cols[0]], new_df[xyz_cols[1]], new_df[xyz_cols[2]] = helix_1(new_df[xyz_cols[0]], new_df[xyz_cols[1]], new_df[xyz_cols[2]])
    if scaling:
        new_df[xyz_cols] = StandardScaler().fit_transform(new_df[xyz_cols])
    return new_df


def transform_2(df, deg):
    new_df = df[["tx", "ty", "tz"]].copy()
    x = df["tx"]
    y = df["ty"]
    z = df["tz"]

    d2 = x ** 2 + y ** 2 + z ** 2
    d = np.sqrt(d2)
    r = np.sqrt(x ** 2 + y ** 2)

    phi = (deg/180.0) * np.pi * z / 2000
    # u = x / d2
    # v = y / d2
    new_df["tx"] = x * np.cos(-phi) - y * np.sin(-phi)
    new_df["ty"] = x * np.sin(-phi) + y * np.cos(-phi)

    return new_df


def transform_3(df):
    x = df["tx"]
    y = df["ty"]
    z = df["tz"]

    phi = 40 / 180 * np.pi * z * 0.0005
    hx = x * np.cos(-phi) - y * np.sin(-phi)
    hy = x * np.sin(-phi) + y * np.cos(-phi)
    hz = z

    new_df = df[["tx", "ty", "tz"]].copy()
    new_df["tx"] = hx
    new_df["ty"] = hy
    new_df["tz"] = hz

    # hd = np.sqrt(hx ** 2 + hy ** 2 + hz ** 2)
    # hr = np.sqrt(hx ** 2 + hy ** 2)
    # new_df["tx"] /= hd
    # new_df["ty"] /= hd
    # new_df["tz"] /= hr

    return new_df

"""
def merge_2_cluster(pred_1, pred_2):
    pred_2 += max(pred_1) + 1  # prevent repeated cluster ids
    all_cluster_id_1 = Counter(pred_1)  # map from cluster_id in pred_1 to the number of its occurrences
    all_cluster_id_2 = Counter(pred_2)  # map from cluster_id in pred_2 to the number of its occurrences
    for cluster_id, nhits in all_cluster_id_2.most_common():
        for hit_id in np.where(pred_2 == cluster_id)[0]:
            cluster_id_temp = pred_1[hit_id]
            if all_cluster_id_1[cluster_id_temp] < nhits:
                pred_1[hit_id] = cluster_id
    return pred_1


def merge_2_cluster(pred_1, pred_2):
    pred_2[pred_2 > 0] += max(pred_1)
    pred_1[pred_1 == 0] += pred_2[pred_1 == 0]
    return pred_1
"""


def run_multiple_cluster(xyz_array, eps=0.00715, n_theta=20):
    df = pd.DataFrame()
    for theta in np.linspace(0.0, 180.0, n_theta):
        print(theta)
        df["hx"], df["hy"], df["hz"] = helix_1(*helix_2(xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2], theta))
        print("running dbscan test")
        test_dbscan((0.004, 0.008, 0.01, 0.02, 0.04), truth.hit_id, df, truth, True)
    return None


# TODO: important parameter
flag_plot = True
if flag_plot:
    print("PLOTTING mode")
else:
    print("CLUSTERING mode")

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

    print(pd.Series(np.sqrt(truth["tx"] ** 2 + truth["ty"] ** 2)).describe())
    print(pd.Series(np.sqrt(truth["tx"] ** 2 + truth["ty"] ** 2+ truth["tz"] ** 2)).describe())

    if flag_plot:
        plot_track_3d(
            truth,
            # truth.loc[(truth.tpz > 2), :],
            transformer_list=[
                # lambda df: transform_dummy(df, scaling=False),
                lambda df: transform_1(df, scaling=True),
                # transform_dummy,
                # transform_1
            ],
            # clusterer_list=[DBSCAN(eps=0.01, min_samples=1, algorithm='auto', n_jobs=-1)],
            n_tracks=150, cutoff=3)
    else:
        run_multiple_cluster(truth[["tx", "ty", "tz"]].values, n_theta=50)

        # test_dbscan(
        # (0.001, 0.003, 0.008, 0.01, 0.02, 0.03, 0.07, 0.1, 0.3),
        # (0.01, 0.03, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        # hit_id=truth.hit_id, data=transform_3(truth), truth=truth, scaling=False)

