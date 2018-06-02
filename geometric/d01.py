"""
d01.py

explore geometric properties of particle tracks

by Tianyi Miao

Notes:
h1: `rc=np.sqrt(x*x+y*y)` is increasing with respect to `np.abs(z)` within a track
This is true for 80% of tracks, but particles with low momentum are a notable exception

h2: the monotonicity score (spearman) described in h1 is related to az
ac = np.arctan2(y, x)
az = np.arctan2(rc, z)  # the angle between [a ray from origin to the point] and [z-axis]
"""

from utils.session import Session

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def subroutine_1(df):
    """
    explore track size
    """
    track_size = df.groupby("particle_id")["x"].agg("count")
    track_size.drop(0, axis=0, inplace=True)
    plt.hist(track_size, alpha=0.3, color="g", range=(0.5, 21.5), bins=21)
    plt.xticks(range(0, 21))


def subroutine_2(df, n=20):
    """
    plot 3d tracks
    """
    # particle id -> track size
    track_size = df.groupby("particle_id", sort=False)["x"].agg("count")
    # ignore noisy track
    track_size.drop(0, axis=0, inplace=True)
    # ignore small tracks (track size <= 3)
    particle_list = track_size[track_size > 3].index
    # randomly select n particles/tracks
    particle_list = np.random.choice(particle_list, size=n, replace=False)
    # get boolean mask
    selected_idx = df.particle_id.isin(particle_list)
    # aggregate selected df by particle id
    df_agg = df.loc[selected_idx, :].groupby("particle_id", sort=False)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def plot_track(sub_df):
        sub_df = sub_df.sort_values(by="z")
        ax.plot(sub_df.x.values, sub_df.y.values, sub_df.z.values, ".-")
        return sub_df

    noise_df = df.loc[df.particle_id == 0].sample(n=200, replace=False)
    ax.plot(noise_df.x.values, noise_df.y.values, noise_df.z.values, ".", color="grey")
    ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
    plt.show()


def subroutine_3(df):
    def calculate_momentum(sub_df):
        return np.sqrt(sub_df.tpx ** 2 + sub_df.tpy ** 2 + sub_df.tpz ** 2)

    df["tp"] = calculate_momentum(df)
    # print(df[["tpx", "tpy", "tpz", "tp"]].describe())

    """
    h1: `rc=np.sqrt(x*x+y*y)` is increasing with respect to `np.abs(z)` within a track
    """
    # particle id -> track size
    track_size = df.groupby("particle_id", sort=False)["x"].agg("count")
    # ignore noisy track
    track_size.drop(0, axis=0, inplace=True)
    # ignore small tracks (track size <= 3)
    particle_list = track_size[track_size > 3].index
    # only consider the meaningful particles (size > 3 and not noisy)
    df = df.loc[df.particle_id.isin(particle_list), :]

    df.loc[:, "rc"] = np.sqrt(df.x ** 2 + df.y ** 2)
    df.loc[:, "abs_z"] = np.abs(df.z)
    df.loc[:, "az"] = np.arctan2(df.rc, df.z)

    df_agg = df.groupby("particle_id", sort=False)

    def check_monotonic(sub_df):
        """
        returns the spearman correlation coefficient and the size of the data
        """
        spearman_corr = sub_df["rc"].corr(sub_df["abs_z"], method="spearman")
        return pd.Series(
            (spearman_corr,
             sub_df["x"].count(),
             np.mean(calculate_momentum(sub_df)),
             np.mean(sub_df.z),
             np.std(sub_df.z)
             )
        )

    # spearman correlation coefficient between rc and abs_z
    res = df_agg.apply(check_monotonic)
    res.columns = ["spearman", "track_size", "momentum", "mean_z", "std_z"]
    # print("number of tracks violating h1: ", sum(res.spearman < 1.0 - 1e-12))

    # ill_behaved_particles = res[res.spearman < 1.0 - 1e-12].index
    ill_behaved_particles = res[res.spearman < 0.5].index

    if True:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.scatter(x=res.mean_z, y=res.spearman)
        ax1.set_xlabel("mean z in track")
        ax1.set_ylabel("spearman between rc and abs_z")
        ax2 = fig.add_subplot(122)
        ax2.scatter(x=res.std_z, y=res.spearman)
        ax2.set_xlabel("standard deviation of z in track")
        ax2.set_ylabel("spearman between rc and abs_z")
    if False:
        print("total weight for ill-behaved tracks: ", df.weight[df.particle_id.isin(ill_behaved_particles)].sum())
    if False:
        print("number of ill-behaved tracks", len(ill_behaved_particles))
        print("total number of tracks: ", res.shape[0])
    if False:
        plt.hist(res["spearman"])  # plot histogram
    if False:
        plt.scatter(x=np.log1p(res.momentum), y=res.spearman)
        plt.xlabel("particle momentum (log scale)")
        plt.ylabel("spearman between rc and abs_z")
    plt.show()
    if False:
        # visualize tracks from ill-behaved particles
        n = 40
        # select a random subset of particles for visualization
        particle_list = np.random.choice(ill_behaved_particles, size=n, replace=False)
        # get boolean mask
        selected_idx = df.particle_id.isin(particle_list)
        # aggregate selected df by particle id
        df_agg = df.loc[selected_idx, :].groupby("particle_id", sort=False)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        def plot_track(sub_df):
            sub_df = sub_df.sort_values(by="z")
            ax.plot(sub_df.x.values, sub_df.y.values, sub_df.z.values, ".-")
            return sub_df

        df_agg.apply(plot_track)

        plt.show()
    print("")

if __name__ == "__main__":
    print("start running script d01.py")

    s1 = Session(parent_dir="E:/TrackMLData/")
    for hits, truth in s1.remove_train_events(n=10, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        print("=" * 120)
        hits = hits.merge(truth, how="left", on="hit_id")
        subroutine_3(hits)
