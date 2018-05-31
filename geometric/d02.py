"""
d02.py
by Tianyi Miao

Explore the possibility of cone scanning

az margin is set to 2
"""

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import dbscan, DBSCAN

from geometric.session import Session
from geometric.utils import label_encode, reassign_noise, track_completeness, hit_completeness
from trackml.score import score_event


def subroutine_1(df):
    # feature engineering
    df.loc[:, "rc"] = np.sqrt(df.x ** 2 + df.y ** 2)  # radius in cylindrical coordinate
    df.loc[:, "rs"] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)  # radius in spherical coordinate
    df.loc[:, "ac"] = np.arctan2(df.y, df.x)  # from -pi to pi
    df.loc[:, "az"] = np.arctan2(df.rc, df.z)  # from 0 to pi

    track_size = df.groupby("particle_id")["x"].agg("count")  # pd.Series; track id -> track size

    # display track completeness and hit completeness
    step_az = np.pi / 20
    for az_0 in np.arange(0.0, np.pi, step_az):
        print("[{:.4f}, {:.4f}) ".format(az_0, az_0 + step_az), end="")
        print("completeness = {:.5f}".format(
            # hit_completeness(df, (df.az >= az_0) & (df.az < az_0 + step_az), track_size)
            track_completeness(df, (df.az >= az_0) & (df.az < az_0 + step_az))
        ))


def subroutine_2(df):
    # feature engineering
    df.loc[:, "rc"] = np.sqrt(df.x ** 2 + df.y ** 2)  # radius in cylindrical coordinate
    df.loc[:, "rs"] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)  # radius in spherical coordinate
    df.loc[:, "ac"] = np.arctan2(df.y, df.x)  # from -pi to pi
    df.loc[:, "az"] = np.arctan2(df.rc, df.z)  # from 0 to pi

    # useful stats
    track_size = df.groupby("particle_id")["x"].agg("count")  # pd.Series; track id -> track size
    if False:
        for az_center in np.arange(0, 180):
            print("center=" + str(az_center).rjust(3) + ": ", end="")
            for az_margin in np.arange(0, 10, 0.5):
                lo, hi = np.deg2rad(az_center - az_margin), np.deg2rad(az_center + az_margin)
                idx = (df.az >= lo) & (df.az < hi)
                hc = hit_completeness(df, idx, track_size)
                print("{:.1f}: {:.4f}".format(az_margin, hc), end=", ")
            print()

    for az_center in np.arange(0, 180, 2):
        # az_center = 75
        az_margin = 2
        lo, hi = np.deg2rad(az_center - az_margin), np.deg2rad(az_center + az_margin)
        idx = (df.az >= lo) & (df.az < hi)

        print("[{:.4f}, {:.4f}] ".format(lo, hi),
              str(idx.sum()).rjust(5),
              " {:.6f}".format(hit_completeness(df, idx, track_size)))
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(df.loc[idx, "x"].values, df.loc[idx, "y"].values, df.loc[idx, "z"].values, ".")
    plt.show()
    """


def subroutine_3(df):
    res = df.groupby("particle_id")["z"].agg(["min", "max"]).values
    idx = (res[:, 0] < 0) & (res[:, 1] > 0)  # min z < 0 and max z > 0, the track crosses the xy-plane
    print(np.sum(idx), len(res))
    plt.hist2d(x=res[idx, 0], y=res[idx, 1], bins=50)
    plt.xlabel("minimum z"), plt.ylabel("maximum z")

    plt.show()


if __name__ == "__main__":
    print("start running script d01.py")

    s1 = Session(parent_dir="E:/TrackMLData/")
    for hits, truth in s1.remove_train_events(n=10, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        print("=" * 120)
        hits = hits.merge(truth, how="left", on="hit_id")
        subroutine_3(hits)