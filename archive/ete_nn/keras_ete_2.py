"""
try to minimize the expected loss for the following:
train_nn(nn_list_basic, get_feature(hits, theta=np.random.rand()*2*np.pi, flip=np.random.rand()<0.5), permute_target(fy), basic_trainable=True, epochs=..., batch_size=...)
"""
from keras.utils import multi_gpu_model

import numpy as np
import pandas as pd
import os

import keras
from keras.layers import Dense
from keras.models import Model
from keras.models import load_model

from utils.session import Session
import archive.ete_nn.model as myModel

os.chdir("/rscratch/xuanyu/Kaggle-TrackML/")

def _get_quadratic_features(df):
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
    return (_get_quadratic_features(df) if quadratic else df).values


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


def train_nn(nn_list, train_x, train_y, basic_trainable=True, epochs=10, batch_size=4096, verbose=0):
    for layer in nn_list:
        layer.trainable = basic_trainable
    #print(f"shape of fx: {train_x.shape}")
    #print(f"shape of fy: {train_y.shape}")

    tensorboard = keras.callbacks.TensorBoard(log_dir='logs/')
     
    n_targets = train_y.shape[1]
    output_layer = Dense(n_targets, activation="softmax", trainable=True)(nn_list[-1])
    if os.listdir("./checkpoint/") != []:
        print("Model present, loading model")
        temp_model = load_model("./checkpoint/mymodel.h5")
    else:
        print("Model not present, creating model")
        temp_model = Model(inputs=nn_list[0], outputs=output_layer)

    adam = keras.optimizers.adam(lr=0.001)
    parallel_model = multi_gpu_model(temp_model, gpus=6)
    parallel_model.compile(loss='categorical_crossentropy',
                       optimizer=adam)

	# This `fit` call will be distributed on 8 GPUs.
	# Since the batch size is 256, each GPU will process 32 samples.
	#parallel_model.fit(x, y, epochs=20, batch_size=256)
    #temp_model.compile(optimizer=adam, loss="categorical_crossentropy")
	

    #with tf.device('/gpu:0'):
    history = parallel_model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[tensorboard])
    losses = history.history['loss']
    return int(losses[len(losses)-1]), parallel_model

def main():
    print("start running basic neural network")
    np.random.seed(1)  # restart random number generator
    s1 = Session(parent_dir="/rscratch/xuanyu/Kaggle-TrackML/portable-dataset/")
    n_events = 200
    count = 0
    nn_list_basic = myModel.MLP(9)

    for hits_train, truth_train in s1.get_train_events(n=n_events, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        count += 1
        #print(f"{count}/{n_events}")
        hits_train = join_hits_truth(hits_train, truth_train)
        fy = get_target(hits_train)

        loss_global = 5000
        # fx = get_feature(hits, 0.0, flip=False, quadratic=True)
        for i in range(100):
            print("Step: " + str(i))
            loss, model = train_nn(nn_list_basic, get_feature(hits_train, theta=np.random.rand() * 2 * np.pi, flip=np.random.rand() < 0.5, quadratic=True), permute_target(fy),
            basic_trainable=True, epochs=20, batch_size=4096, verbose=1)
            if(loss<loss_global):
                print("Epoch result better than the best, saving model")
                model.save("./checkpoint/"+"mymodel.h5")
            # train_nn(nn_list_basic, fx, permute_target(fy), basic_trainable=True, epochs=4, batch_size=128, verbose=1)


if __name__ == "__main__":
    main()
