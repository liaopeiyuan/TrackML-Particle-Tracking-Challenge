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


def get_lr(sub_df):
    m = linear_model.LinearRegression(fit_intercept=True, normalize=False)
    x_cols = ["z", "z2", "sinz"]
    y_cols = ["x", "y"]
    m.fit(sub_df[x_cols], sub_df[y_cols])
    return m.score(sub_df[x_cols], sub_df[y_cols])


"""
def get_lr_weight(sub_df, cols=("x", "y", "x2", "y2", "z2", "xy", "xz", "yz")):
    m = linear_model.LinearRegression(fit_intercept=True, normalize=False)
    # m.fit(sub_df, np.ones(sub_df.shape[0]))
    # return mean_squared_error(y_true=np.ones(sub_df.shape[0]), y_pred=m.predict(sub_df))
    m.fit(sub_df[list(cols)], sub_df["z"])
    return pd.Series(np.hstack((m.intercept_, m.coef_)), index=["00"] + list(cols))
    # return m.score(sub_df[cols], sub_df["z"])
    # return mean_squared_error(y_true=sub_df["z"], y_pred=m.predict(sub_df[["x", "y"]]))


def get_lr_r2(sub_df, cols=("x", "y", "x2", "y2", "z2", "xy", "xz", "yz")):
    m = linear_model.LinearRegression(fit_intercept=True, normalize=False)
    m.fit(sub_df[list(cols)], sub_df["z"])
    return m.score(sub_df[list(cols)], sub_df["z"])
"""


def plot_cdf(data, bins=100):
    values, base = np.histogram(data, bins=bins)
    cumulative = np.cumsum(values) / len(data)
    plt.plot(base[:-1], cumulative)


def plot_track_subroutine(sub_df, ax):
    sub_df = sub_df.sort_values(by="z")
    ax.plot(sub_df["x"].values, sub_df["y"].values, sub_df["z"].values, ".-")
    return sub_df


def get_hits_df(hits_df, truth_df):
    # get a perfect dataset for clustering
    df = truth_df[["hit_id", "particle_id"]].merge(hits_df[["hit_id", "x", "y", "z"]], on="hit_id")
    df.drop("hit_id", axis=1, inplace=True)
    df = df.loc[df["particle_id"] != 0]
    # remove small tracks
    df = df.merge(pd.DataFrame(df.groupby("particle_id").size().rename("track_size")), left_on="particle_id", right_index=True)
    df = df.loc[df["track_size"] > 3]
    # scale coordinate values
    # hits[["x", "y", "z"]] = scale(hits[["x", "y", "z"]])
    df = get_quadric_features(df)
    return df


def run_sin_score(hits_df, n):
    hits_df = hits_df.copy()
    hits_df["sinz"] = np.sin(hits_df["z"] / n * np.pi)
    r2 = hits_df.groupby("particle_id").apply(lambda x: get_lr(x))
    return r2.sum()


def main_1():
    print("start running clustering and regression")
    np.random.seed(1)  # restart random number generator
    s1 = Session(parent_dir="E:/TrackMLData/")
    n_events = 4

    all_cols = ["x", "y", "x2", "y2", "z2", "xy", "xz", "yz"]
    selected_cols = ["x", "y", "x2", "y2", "z2", "xy", "xz", "yz"]

    tau = 0.9

    count = 0
    for hits, truth in s1.get_train_events(n=n_events, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        count += 1
        print(count)

        hits = get_hits_df(hits, truth)

        # get new features for helix
        hits["sinz"] = np.sin(hits["z"] / 1000 * np.pi)

        # weight = hits.groupby("particle_id").apply(lambda x: get_lr_weight(x, selected_cols))
        r2 = hits.groupby("particle_id").apply(lambda x: get_lr(x))
        print(f"proportion of perfect explanations: {(r2 > tau).sum()}/{hits.particle_id.nunique()}")
        # plt.hist(r2, bins=2000)
        # plt.show()

        # prepare for 3d plotting
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        for p_id in np.random.choice(r2[r2 < tau].index, size=min((r2 < tau).sum(), 30), replace=False):
            plot_track_subroutine(hits.loc[hits["particle_id"] == p_id], ax)

        plt.show()
        # print("preparing scatter matrix")
        # from pandas.plotting import scatter_matrix
        # scatter_matrix(weight)
        # plt.show()

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


def main_2():
    from bayes_opt import BayesianOptimization as BO

    s1 = Session(parent_dir="E:/TrackMLData/")
    train_list = [get_hits_df(hits, truth) for hits, truth in s1.get_train_events(n=10, content=[s1.HITS, s1.TRUTH], randomness=True)[1]]


    sin_bo_1 = BO(lambda n: sum(run_sin_score(hits_df, n) for hits_df in train_list), pbounds={"n": (0, 10000)}, verbose=1)
    sin_bo_1.maximize(init_points=5, n_iter=25, kappa=5)


if __name__ == "__main__":
    main_2()
