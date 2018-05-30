"""
d02.py
by Tianyi Miao

Explore the possibility of cone scanning
"""

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from geometric.session import Session
from trackml.score import score_event


def hit_completeness(df, idx, track_size):
    """
    (the number of non-noisy hits in the idx) / (the total number of hits from all particles
    that have at least 1 hit in the idx)
    """
    num = (df.loc[idx, "particle_id"] != 0).sum()
    all_particles = df.loc[idx, "particle_id"].unique().tolist()
    all_particles.remove(0)
    denom = track_size[all_particles].sum()
    return num / denom


def track_completeness(df, idx):
    """
    (number of tracks with all hits in the region) / (number of tracks that have at least 1 hit in the region)
    """
    all_particles = df.loc[idx, "particle_id"].unique().tolist()
    all_particles.remove(0)

    agg_1 = df.loc[idx, :].groupby("particle_id", sort=True)["x"].agg("count").drop(0, inplace=False)
    agg_2 = df.loc[df.particle_id.isin(all_particles), :].groupby("particle_id", sort=True)["x"].agg("count")
    return np.mean(agg_1 == agg_2)


def subroutine_1(df):
    # feature engineering
    df.loc[:, "rc"] = np.sqrt(df.x ** 2 + df.y ** 2)  # radius in cylindrical coordinate
    df.loc[:, "rs"] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)  # radius in spherical coordinate
    df.loc[:, "ac"] = np.arctan2(df.y, df.x)  # from -pi to pi
    df.loc[:, "az"] = np.arctan2(df.rc, df.z)  # from 0 to pi

    track_size = df.groupby("particle_id")["x"].agg("count")  # pd.Series; track id -> track size

    step_az = np.pi / 20
    for az_0 in np.arange(0.0, np.pi, step_az):
        print("[{:.4f}, {:.4f}) ".format(az_0, az_0 + step_az), end="")
        print("completeness = {:.5f}".format(
            # hit_completeness(df, (df.az >= az_0) & (df.az < az_0 + step_az), track_size)
            track_completeness(df, (df.az >= az_0) & (df.az < az_0 + step_az))
        ))


if __name__ == "__main__":
    print("start running script d01.py")

    s1 = Session(parent_dir="E:/TrackMLData/")
    for hits, truth in s1.remove_train_events(n=10, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        print("=" * 120)
        hits = hits.merge(truth, how="left", on="hit_id")
        subroutine_1(hits)