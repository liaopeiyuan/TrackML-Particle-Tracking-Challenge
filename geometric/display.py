import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd


def plot_track_3d(df, transformer_list, clusterer_list=(), n_tracks=20, cutoff=3, verbose=False):
    """
    :param df: Pandas DataFrame containing hit coordinate information
    must have the following columns: particle_id, x, y, z

    :param transformer_list: list of transformer functions.
    transformer function: df -> a numpy array of shape [n_samples, 3]

    :param clusterer_list: a list of clusterer objects; each of them will be used to transform the new dataset
    example: [DBSCAN(eps=0.01), DBSCAN(eps=0.1)]

    :param n_tracks: the number of particles/tracks to display

    :param cutoff: particles with less than or equal to cutoff hits will be ignored.

    :param verbose: if set to True, print the particle id and the coordinate matrix in the console.
    """

    for c in ["x", "y", "z", "particle_id"]:
        if c not in df.columns:
            raise ValueError("[plot_track_3d] input DataFrame does not contain {}".format(c))

    df = df.copy()

    def plot_track(sub_df):
        sub_df = sub_df.sort_values(by="z")
        ax.plot(sub_df.v1.values, sub_df.v2.values, sub_df.v3.values, ".-")
        return sub_df

    track_size = df.groupby("particle_id", sort=False)["x"].agg("count")  # particle id -> track size
    track_size.drop(0, axis=0, inplace=True)  # ignore noisy track
    particle_list = track_size[track_size > cutoff].index  # ignore small tracks (track size <= cutoff)
    particle_list = np.random.choice(particle_list, size=min(n_tracks, particle_list.size), replace=False)

    selected_idx = df.particle_id.isin(particle_list)  # get boolean mask

    fig = plt.figure()
    i = 1

    for transformer in transformer_list:  # iterate over transformers
        ax = fig.add_subplot(len(transformer_list), len(clusterer_list) + 1, i, projection='3d')
        # transform the dataset using transformer
        new_array = transformer(df)
        df.loc[:, "v1"] = new_array[:, 0]
        df.loc[:, "v2"] = new_array[:, 1]
        df.loc[:, "v3"] = new_array[:, 2]

        df_agg = df.loc[selected_idx, :].groupby("particle_id", sort=False)  # aggregate selected df by particle id
        df_agg.apply(plot_track)
        ax.set_xlabel("v1"), ax.set_ylabel("v2"), ax.set_zlabel("v3")

        i += 1  # increment the subplot index

        for clusterer in clusterer_list:
            ax = fig.add_subplot(len(transformer_list), len(clusterer_list) + 1, i, projection='3d')
            df.loc[:, "cluster_id"] = clusterer.fit_predict(new_array)
            df_agg = df.loc[selected_idx, :].groupby("cluster_id", sort=False)  # aggregate selected df by particle id
            df_agg.apply(plot_track)
            ax.set_xlabel("v1"), ax.set_ylabel("v2"), ax.set_zlabel("v3")
            i += 1
    plt.show()


def plot_track_fast(df, transformer_list, n_tracks=20, cutoff=3):
    df = df.copy()

    def plot_track(sub_df):
        sub_df = sub_df.sort_values(by="z")
        ax.plot(sub_df.v1.values, sub_df.v2.values, sub_df.v3.values, ".-")
        return sub_df

    track_size = df.groupby("particle_id", sort=False)["x"].agg("count")  # particle id -> track size
    track_size.drop(0, axis=0, inplace=True)  # ignore noisy track
    particle_list = track_size[track_size > cutoff].index  # ignore small tracks (track size <= cutoff)
    particle_list = np.random.choice(particle_list, size=min(n_tracks, particle_list.size), replace=False)

    selected_idx = df.particle_id.isin(particle_list)  # get boolean mask

    for transformer in transformer_list:  # iterate over transformers
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # transform the dataset using transformer
        new_array = transformer(df)
        df.loc[:, "v1"] = new_array[:, 0]
        df.loc[:, "v2"] = new_array[:, 1]
        df.loc[:, "v3"] = new_array[:, 2]

        df_agg = df.loc[selected_idx, :].groupby("particle_id", sort=False)  # aggregate selected df by particle id
        df_agg.apply(plot_track)
        ax.set_xlabel("v1"), ax.set_ylabel("v2"), ax.set_zlabel("v3")

        plt.show()
