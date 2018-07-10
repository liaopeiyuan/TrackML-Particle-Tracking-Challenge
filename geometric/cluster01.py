"""
regression and clustering
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import scale

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


def get_lr_result(sub_df, cols=("x", "y", "x2", "y2", "z2", "xy", "xz", "yz")):
    m = linear_model.LinearRegression(fit_intercept=True, normalize=False)
    # m.fit(sub_df, np.ones(sub_df.shape[0]))
    # return mean_squared_error(y_true=np.ones(sub_df.shape[0]), y_pred=m.predict(sub_df))
    m.fit(sub_df[list(cols)], sub_df["z"])
    return pd.Series(np.hstack((m.intercept_, m.coef_)), index=["00"] + list(cols))
    # return m.score(sub_df[cols], sub_df["z"])
    # return mean_squared_error(y_true=sub_df["z"], y_pred=m.predict(sub_df[["x", "y"]]))


def plot_cdf(data, bins=100):
    values, base = np.histogram(data, bins=bins)
    cumulative = np.cumsum(values) / len(data)
    plt.plot(base[:-1], cumulative)


def plot_track_subroutine(sub_df):
    sub_df = sub_df.sort_values(by="z")
    plt.plot(sub_df.v1.values, sub_df.v2.values, sub_df.v3.values, ".-")
    return sub_df


if __name__ == "__main__":
    print("start running clustering and regression")
    np.random.seed(1)  # restart random number generator
    s1 = Session(parent_dir="E:/TrackMLData/")
    n_events = 1

    # prepare for 3d plotting
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    from itertools import chain, combinations

    all_cols = ["x", "y", "x2", "y2", "z2", "xy", "xz", "yz"]

    record_1 = {}
    for selected_cols in chain(combinations(all_cols, r=len(all_cols)), combinations(all_cols, r=len(all_cols)-1)):
        record_1[frozenset(selected_cols)] = []

    count = 0
    for hits, truth in s1.get_train_events(n=n_events, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        count += 1
        print(count)
        hits = truth[["hit_id", "particle_id"]].merge(hits[["hit_id", "x", "y", "z"]], on="hit_id")
        hits.drop("hit_id", axis=1, inplace=True)
        hits = hits.loc[hits["particle_id"] != 0]
        # remove small tracks
        hits = hits.merge(pd.DataFrame(hits.groupby("particle_id").size().rename("track_size")), left_on="particle_id", right_index=True)
        hits = hits.loc[hits["track_size"] > 3]

        # scale coordinate values
        # hits[["x", "y", "z"]] = scale(hits[["x", "y", "z"]])
        hits = get_quadric_features(hits)
        res = hits.groupby("particle_id").apply(get_lr_result)

        print("preparing scatter matrix")
        from pandas.plotting import scatter_matrix
        scatter_matrix(res)
        plt.show()

        # print((res < 0.999).sum())
        # plot_cdf(res, bins=1000)
        # print(res.describe())
        # print(res.loc[res < 0.9999])
        # for wild_id in res.loc[res < 0.9999].index:
        #     sub_df = hits.loc[hits["particle_id"] == wild_id, ["x", "y", "z"]]
        #     sub_df.sort_values(by="z")
        #     ax.plot(sub_df.x.values, sub_df.y.values, sub_df.z.values, ".-")
    plt.show()
    # plt.hist(error_2, bins=100)


