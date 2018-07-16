import numpy as np
import pandas as pd

import keras


def get_quadratic_features(df):
    df["x2"] = df["x"] ** 2
    df["y2"] = df["y"] ** 2
    df["z2"] = df["z"] ** 2
    df["xy"] = df["x"] * df["y"]
    df["xz"] = df["x"] * df["z"]
    df["yz"] = df["y"] * df["z"]
    return df


def get_feature(hits, theta, flip, quadratic=True):
    """
    get the feature array for neural network fitting
    theta: the (radian) angle of rotation around the z axis
    flip: whether flip the points across the xy-plane
    """
    df = hits[["x", "y", "z"]].copy()
    r = np.sqrt(df["x"]**2 + df["y"]**2)
    a = np.arctan2(df["y"], df["x"]) + theta
    df.loc[:, "x"] = np.cos(a) * r
    df.loc[:, "y"] = np.sin(a) * r
    if flip:
        df.loc[:, "z"] = -df["z"]
    return (get_quadratic_features(df) if quadratic else df).values


def get_target(hits):
    hits = hits[["particle_id"]].copy()
    hits = hits.merge(pd.DataFrame(hits.groupby("particle_id").size().rename("track_size")), left_on="particle_id", right_index=True)
    hits.loc[(hits["track_size"] < 4) | (hits["particle_id"] == 0), "particle_id"] = np.nan
    return pd.get_dummies(hits["particle_id"], dummy_na=False).values


def permute_target(target):
    return target[:, np.random.permutation(range(target.shape[1]))]


def join_hits_truth(hits, truth):
    hits = truth[["hit_id", "particle_id"]].merge(hits[["hit_id", "x", "y", "z"]], on="hit_id")
    hits.drop("hit_id", axis=1, inplace=True)
    return hits

