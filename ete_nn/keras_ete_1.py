import numpy as np
import pandas as pd

import keras
from keras.layers import Input, Dense
from keras.models import Model


from utils.session import Session


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


def get_basic_nn():
    nn_dict = {"main_input": Input(shape=(9,), name="main_input")}
    nn_dict["x1"] = Dense(64, activation="relu", name="x1")(nn_dict["main_input"])
    nn_dict["x2"] = Dense(128, activation="relu", name="x2")(nn_dict["x1"])
    nn_dict["x3"] = Dense(256, activation="relu", name="x3")(nn_dict["x2"])
    nn_dict["x4"] = Dense(256, activation="relu", name="x4")(nn_dict["x3"])
    nn_dict["x5"] = Dense(256, activation="relu", name="x5")(nn_dict["x4"])
    nn_dict["x6"] = Dense(128, activation="relu", name="x6")(nn_dict["x5"])
    nn_dict["x7"] = Dense(64, activation="relu", name="x7")(nn_dict["x6"])
    return nn_dict


if __name__ == "__main__":
    print("start running basic neural network")
    np.random.seed(1)  # restart random number generator
    s1 = Session(parent_dir="E:/TrackMLData/")
    n_events = 4
    count = 0
    nn_dict_basic = get_basic_nn()
    for hits, truth in s1.get_train_events(n=n_events, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        count += 1
        print(f"{count}/{n_events}")
        hits = join_hits_truth(hits, truth)
        fx = get_feature(hits, 0.0, flip=False, quadratic=True)
        fy = get_target(hits)
        print(f"shape of fx: {fx.shape}")
        print(f"shape of fy: {fy.shape}")
        n_targets = fy.shape[1]
        output_layer = Dense(n_targets, activation="softmax", name="temp_output")(nn_dict_basic["x7"])
        temp_model = Model(inputs=nn_dict_basic["main_input"], outputs=output_layer)
        temp_model.compile(optimizer="adam", loss="categorical_crossentropy")
        temp_model.fit(fx, fy, batch_size=32, verbose=1, validation_split=0.1)
