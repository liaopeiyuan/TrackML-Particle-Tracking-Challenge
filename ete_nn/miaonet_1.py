import numpy as np
import pandas as pd

from utils.session import Session

from sklearn.preprocessing import LabelEncoder

from keras.models import Model
from keras.layers import Input, Embedding, Dense, PReLU, BatchNormalization


def prepare_df(hits_df, truth_df):
    df = truth_df[["hit_id", "particle_id", "weight"]].merge(hits_df[["hit_id", "x", "y", "z", "volume_id", "layer_id", "module_id"]], on="hit_id")
    # drop useless rows
    df["track_size"] = df.groupby("particle_id")["particle_id"].transform("count")
    # df = df.merge(pd.DataFrame(hits_df.groupby("particle_id").size().rename("track_size")), left_on="particle_id", right_index=True)
    df = df.loc[(df["track_size"] > 3) & (df["particle_id"] != 0), ["x", "y", "z", "volume_id", "layer_id", "module_id", "weight", "particle_id"]]
    # prepare categorical variables for embedding
    # volume_dict = {7: 0, 8: 1, 9: 2, 12: 3, 13: 4, 14: 5, 16: 6, 17: 7, 18: 8}
    # layer_dict = {2: 0, 4: 1, 6: 2, 8: 3, 10: 4, 12: 5, 14: 6}
    # df["volume_id"] = df["volume_id"].map(volume_dict)
    # df["layer_id"] = df["layer_id"].map(layer_dict)
    # df["module_id"] -= 1
    return df


def get_basic_nn(input_size=9):
    nn_list = [Input(shape=(input_size,))]
    for layer in [
        Dense(32), BatchNormalization(), PReLU(),
        Dense(64), BatchNormalization(), PReLU(),
        Dense(128), BatchNormalization(), PReLU(),
        Dense(128), BatchNormalization(), PReLU(),
        Dense(128), BatchNormalization(), PReLU(),
        Dense(128), BatchNormalization(), PReLU(),
        Dense(128), BatchNormalization(), PReLU(),
        Dense(64), BatchNormalization(), PReLU(),
    ]:
        nn_list.append(layer(nn_list[-1]))
    return nn_list


def train_nn(nn_list, fx, fy, basic_trainable=True, epochs=10, batch_size=64, loss="categorical_crossentropy", verbose=1):
    for layer in nn_list:
        layer.trainable = basic_trainable
    print(f"shape of fx: {fx.shape}")
    print(f"shape of fy: {fy.shape}")
    output_layer = Dense(np.max(fy) + 1, activation="softmax", trainable=True)(nn_list[-1])
    temp_model = Model(inputs=nn_list[0], outputs=output_layer)
    temp_model.compile(optimizer="adam", loss=loss)
    temp_model.fit(fx, fy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    

def get_target(df):
    return LabelEncoder().fit_transform(df["particle_id"])


def permute_target(target):
    return np.random.permutation(np.max(target) + 1)[target]

    
def augment_1(df, theta):
    df = df[["x", "y", "z"]].copy()
    r = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    a = np.arctan2(df["y"], df["x"]) + theta
    df.loc[:, "x"] = np.cos(a) * r
    df.loc[:, "y"] = np.sin(a) * r
    return df


def get_feature(df):
    return df
    
if __name__ == '__main__':
    pass
    

"""
s1 = Session("../portable-dataset/")
volume_set = set()
layer_set = set()
module_set = set()
count = 0
for hits, in s1.get_train_events(n=10000, content=[s1.HITS])[1]:
    volume_set.update(hits.volume_id)
    layer_set.update(hits.layer_id)
    module_set.update(hits.module_id)
    if count % 100 == 0:
        print(len(volume_set), len(layer_set), len(module_set))
    count += 1

for hits, in s1.get_test_event(n=10000, content=[s1.HITS])[1]:
    volume_set.update(hits.volume_id)
    layer_set.update(hits.layer_id)
    module_set.update(hits.module_id)
    if count % 100 == 0:
        print(len(volume_set), len(layer_set), len(module_set))
    count += 1
"""
# print(hits.columns.tolist())
# print(truth.columns.tolist())

