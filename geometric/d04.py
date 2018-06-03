"""
d04.py

explore the fundamental properties of helix unrolling

by Tianyi Miao
"""

import numpy as np
import pandas as pd

from geometric.display import plot_track_fast
from utils.session import Session


def subroutine_plot_unroll(df):
    df = df.copy()

    df.loc[:, "r3"] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
    df.loc[:, "rt"] = np.sqrt(df.x ** 2 + df.y ** 2)
    df.loc[:, "a0"] = np.arctan2(df.y, df.x)
    df.loc[:, "z1"] = df.z / df.rt

    df.loc[:, "psi"] = np.arctan2(np.sqrt(df.x ** 2 + df.y ** 2), np.abs(df.z))
    idx = df.psi > np.deg2rad(70)

    class tf1:
        def __init__(self, dz):
            self.dz = dz

        def __call__(self, df):
            df = df[["a0", "r3", "z", "rt"]].copy()
            df.loc[:, "a1"] = df.a0 + self.dz * df.r3
            df.loc[:, "x_new"] = np.cos(df.a1)  # * df.rt
            df.loc[:, "y_new"] = np.sin(df.a1)  # * df.rt
            return df[["x_new", "y_new", "z"]].values

    plot_track_fast(df.loc[idx, :], [tf1(dz) for dz in np.arange(-1e-3, 2e-3, 1e-4)], n_tracks=50)


if __name__ == "__main__":
    print("start running script d04.py; helix unrolling visualization")
    np.random.seed()  # restart random number generator
    s1 = Session(parent_dir="E:/TrackMLData/")
    n_events = 20

    for hits, truth in s1.get_train_events(n=n_events, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        print("=" * 120)
        hits = hits.merge(truth, how="left", on="hit_id")
        subroutine_plot_unroll(hits)

