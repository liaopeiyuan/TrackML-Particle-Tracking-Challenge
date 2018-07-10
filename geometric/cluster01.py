"""
regression and clustering
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

from trackml.score import score_event

from utils.session import Session


def get_quadric_features(df):
    df["x2"] = df["x"] ** 2
    df["y2"] = df["y"] ** 2
    df["z2"] = df["z"] ** 2
    df["xy"] = df["x"] * df["y"]
    df["xz"] = df["x"] * df["z"]
    df["yz"] = df["y"] * df["z"]
    return df


def get_lr_residual(sub_df):
    m = linear_model.LinearRegression(fit_intercept=False, normalize=False)
    m.fit(sub_df, np.ones(sub_df.shape[0]))
    return mean_squared_error(y_true=np.ones(sub_df.shape[0]), y_pred=m.predict(sub_df))


if __name__ == "__main__":
    print("start running clustering and regression")
    np.random.seed()  # restart random number generator
    s1 = Session(parent_dir="E:/TrackMLData/")
    n_events = 2

    for hits, truth in s1.get_train_events(n=n_events, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        hits = truth[["hit_id", "particle_id"]].merge(hits[["hit_id", "x", "y", "z"]], on="hit_id")
        hits.drop("hit_id", axis=1, inplace=True)
        hits = hits.loc[hits["particle_id"] != 0]
        hits = get_quadric_features(hits)
        print("start applying linear regression")
        error_2 = hits.groupby("particle_id").apply(get_lr_residual)
        print(error_2.describe())
        plt.hist(error_2)
        plt.show()

