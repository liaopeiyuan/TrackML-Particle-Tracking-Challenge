"""
d04.py

explore the fundamental properties of helix unrolling

by Tianyi Miao
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from geometric.display import plot_track_fast
from utils.session import Session


def subroutine_plot_unroll(df):
    df = df.copy()

    df.loc[:, "rs"] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)
    df.loc[:, "r3"] = df.rs
    df.loc[:, "rt"] = np.sqrt(df.x ** 2 + df.y ** 2)
    df.loc[:, "a0"] = np.arctan2(df.y, df.x)
    df.loc[:, "z1"] = df.z / df.rt
    # df.loc[:, "z2"] = df.z / df.rs

    df.loc[:, "psi"] = np.arctan2(np.sqrt(df.x ** 2 + df.y ** 2), np.abs(df.z))

    idx = df.psi > np.deg2rad(70)
    # idx = df.psi < np.deg2rad(15)

    class tf1:
        def __init__(self, dz):
            self.dz = dz

        def __call__(self, df):
            # df = df[["a0", "r3", "z", "rt", "z1"]].copy()
            # df.loc[:, "a1"] = df.a0 + self.dz * df.r3
            df.loc[:, "a1"] = df.a0 + self.dz * df.rt
            df.loc[:, "x_new"] = np.cos(df.a1) * df.rt
            df.loc[:, "y_new"] = np.sin(df.a1) * df.rt
            return df[["x_new", "y_new", "z1"]].values

        def __str__(self):
            return "helix unroll dz = {:.4e}".format(self.dz)

    plot_track_fast(df.loc[idx, :], [tf1(dz) for dz in np.arange(-1e-3, 2e-3, 1e-4)], n_tracks=50)


def subroutine_plot_z1_range(df):
    idx = df.particle_id != 0

    df.loc[:, "psi"] = np.arctan2(np.sqrt(df.x ** 2 + df.y ** 2), np.abs(df.z))
    df.loc[:, "rc"] = np.sqrt(df.x ** 2 + df.y ** 2)
    df.loc[:, "z1"] = df.z / df.rc
    df_agg = df.loc[idx, :].groupby("particle_id", sort=True)

    def get_z1_stats(sub_df):
        return sub_df.z1.max() - sub_df.z1.min()

    def get_psi_stats(sub_df):
        return sub_df.psi.max() - sub_df.psi.min()

    plt.hist2d(x=df_agg.apply(get_psi_stats).values, y=df_agg.apply(get_z1_stats).values, bins=20, normed=True)
    plt.xlabel("psi range")
    plt.ylabel("z1 range")
    plt.show()


if __name__ == "__main__":
    print("start running script d04.py; helix unrolling visualization")
    np.random.seed()  # restart random number generator
    s1 = Session(parent_dir="E:/TrackMLData/")
    n_events = 20

    for hits, truth in s1.get_train_events(n=n_events, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        print("=" * 120)
        hits = hits.merge(truth, how="left", on="hit_id")
        subroutine_plot_z1_range(hits)
        # subroutine_plot_unroll(hits)

