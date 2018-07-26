
# coding: utf-8

# In[21]:


import numpy as np
import pandas as pd
import os

import keras
from keras.layers import Input, Dense, BatchNormalization, Dropout, PReLU
from keras.models import Model
from keras.models import load_model

from time import time

from keras.models import Model, load_model, Sequential
from keras.layers import Input, BatchNormalization, Add, Activation,Dense, PReLU, Dropout, Flatten,concatenate, Reshape
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras import backend as K
from keras.layers import Activation


# In[22]:


from trackml.dataset import load_event

class Session(object):
    """
    A highly integrated framework for efficient data loading, prediction submission, etc. in TrackML Challenge
    (improved version of the official TrackML package)

    Precondition: the parent directory must be organized as follows:
    - train (directory)
        - event000001000-cells.csv
        ...
    - test (directory)
        - event000000001-cells.csv
        ...
    - detectors.csv
    - sample_submission.csv
    """
    # important constants to avoid spelling errors
    HITS = "hits"
    CELLS = "cells"
    PARTICLES = "particles"
    TRUTH = "truth"

    def __init__(self, parent_dir="./", train_dir="train/", test_dir="test/", detectors_dir="detectors.csv",
                 sample_submission_dir="sample_submission.csv"):
        """
        default input:
        Session("./", "train/", "test/", "detectors.csv", "sample_submission.csv")
        Session(parent_dir="./", train_dir="train/", test_dir="test/", detectors_dir="detectors.csv", sample_submission_dir="sample_submission.csv")
        """
        self._parent_dir = parent_dir
        self._train_dir = train_dir
        self._test_dir = test_dir
        self._detectors_dir = detectors_dir
        self._sample_submission_dir = sample_submission_dir

        if not os.path.isdir(self._parent_dir):
            raise ValueError("The input parent directory {} is invalid.".format(self._parent_dir))

        # there are 8850 events in the training dataset; some ids from 1000 to 9999 are skipped
        if os.path.isdir(self._parent_dir + self._train_dir):
            self._train_event_id_list = sorted(
                set(int(x[x.index("0"):x.index("-")]) for x in os.listdir(self._parent_dir + self._train_dir)))
        else:
            self._train_dir = None
            self._train_event_id_list = []

        if os.path.isdir(self._parent_dir + self._test_dir):
            self._test_event_id_list = sorted(
                set(int(x[x.index("0"):x.index("-")]) for x in os.listdir(self._parent_dir + self._test_dir)))
        else:
            self._test_dir = None
            self._test_event_id_list = []

        if not os.path.exists(self._parent_dir + self._detectors_dir):
            self._detectors_dir = None

        if not os.path.exists(self._parent_dir + self._sample_submission_dir):
            self._sample_submission_dir = None

    @staticmethod
    def get_event_name(event_id):
        return "event" + str(event_id).zfill(9)

    def get_train_events(self, n=10, content=(HITS, TRUTH), randomness=True):
        n = min(n, len(self._train_event_id_list))
        if randomness:
            event_ids = np.random.choice(self._train_event_id_list, size=n, replace=False).tolist()
        else:
            event_ids, = self._train_event_id_list[:n]
            self._train_event_id_list = self._train_event_id_list[n:] + self._train_event_id_list[:n]

        event_names = [Session.get_event_name(event_id) for event_id in event_ids]
        return event_names,             (load_event(self._parent_dir + self._train_dir + event_name, content) for event_name in event_names)

    def remove_train_events(self, n=10, content=(HITS, TRUTH), randomness=True):
        """
        get n events from self._train_event_id_list:
        if random, get n random events; otherwise, get the first n events
        :return:
         1. ids: event ids
         2. an iterator that loads a tuple of hits/cells/particles/truth files
        remove these train events from the current id list
        """
        n = min(n, len(self._train_event_id_list))
        if randomness:
            event_ids = np.random.choice(self._train_event_id_list, size=n, replace=False).tolist()
            for event_id in event_ids:
                self._train_event_id_list.remove(event_id)
        else:
            event_ids, self._train_event_id_list = self._train_event_id_list[:n], self._train_event_id_list[n:]

        event_names = [Session.get_event_name(event_id) for event_id in event_ids]
        return event_names,             (load_event(self._parent_dir + self._train_dir + event_name, content) for event_name in event_names)

    def get_test_event(self, n=3, content=(HITS, TRUTH), randomness=True):
        n = min(n, len(self._test_event_id_list))
        if randomness:
            event_ids = np.random.choice(self._test_event_id_list, size=n, replace=False).tolist()
        else:
            event_ids, = self._test_event_id_list[:n]
            self._test_event_id_list = self._test_event_id_list[n:] + self._test_event_id_list[:n]

        event_names = [Session.get_event_name(event_id) for event_id in event_ids]
        return event_names,             (load_event(self._parent_dir + self._test_dir + event_name, content) for event_name in event_names)
    
    def remove_test_events(self, n=10, content=(HITS, CELLS), randomness=False):
        n = min(n, len(self._test_event_id_list))
        if randomness:
            event_ids = np.random.choice(self._test_event_id_list, size=n, replace=False).tolist()
            for event_id in event_ids:
                self._test_event_id_list.remove(event_id)
        else:
            event_ids, self._test_event_id_list = self._test_event_id_list[:n], self._test_event_id_list[n:]
        event_names = [Session.get_event_name(event_id) for event_id in event_ids]
        return event_names,             (load_event(self._parent_dir + self._test_dir + event_name, content) for event_name in event_names)

    def make_submission(self, predictor, path):
        """
        :param predictor: function, predictor(hits: pd.DataFrame, cells: pd.DataFrame)->np.array
         takes in hits and cells data frames, return a numpy 1d array of cluster ids
        :param path: file path for submission file
        """
        sub_list = []  # list of predictions by event
        for event_id in self._test_event_id_list:
            event_name = Session.get_event_name(event_id)

            hits, cells = load_event(self._parent_dir + self._test_dir + event_name, (Session.HITS, Session.CELLS))
            pred = predictor(hits, cells)  # predicted cluster labels
            sub = pd.DataFrame({"hit_id": hits.hit_id, "track_id": pred})
            sub.insert(0, "event_id", event_id)
            sub_list.append(sub)
        final_submission = pd.concat(sub_list)
        final_submission.to_csv(path, sep=",", header=True, index=False)


# In[26]:


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


def train_nn(nn_list, train_x, train_y, basic_trainable=True, epochs=10, batch_size=64, verbose=0):
    for layer in nn_list:
        layer.trainable = basic_trainable
    print(f"shape of fx: {train_x.shape}")
    print(f"shape of fy: {train_y.shape}")

    tensorboard = keras.callbacks.TensorBoard(log_dir='logs/')
    early_stopping = EarlyStopping(patience=50, verbose=1)
    model_checkpoint = ModelCheckpoint("./keras.model", save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=10, min_lr=0.00001, verbose=1)
     
    n_targets = train_y.shape[1]
    output_layer = Dense(n_targets, activation="softmax", trainable=True)(nn_list[-1])
    if os.listdir("./checkpoint/") != []:
        print("Model present, loading model")
        temp_model = load_model("./checkpoint/mymodel.h5")
    else:
        print("Model not present, creating model")
        temp_model = Model(inputs=nn_list[0], outputs=output_layer)

    adam = keras.optimizers.adam(lr=0.001)
    temp_model.compile(optimizer=adam, loss="categorical_crossentropy")
    history = temp_model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[tensorboard])

    losses = history.history['loss']
    return int(losses[len(losses)-1]), temp_model


# In[34]:


def MLP_with_dropout(input_size=9, rate=0.5):
    nn_list = [Input(shape=(input_size,))]
    for layer in[
        Dense(32), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(64), BatchNormalization(), PReLU(),
        Dense(128), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(256), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(512), BatchNormalization(), PReLU(),
        Dense(1024), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(2048), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(4096), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(512), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(128), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(64), BatchNormalization(), PReLU(),
    ]:
      nn_list.append(layer(nn_list[-1]))
    return nn_list  

def MLP(input_size=9, rate=0.5):
    nn_list = [Input(shape=(input_size,))]
    for layer in[
        Dense(32, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        #Dropout(rate),
        Dense(64, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(128, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(128, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(128, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(128, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        #Dropout(rate),
        Dense(256, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(256, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(256, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(512, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(512, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(512, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(512, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dropout(rate),
        Dense(1024, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(1024, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
        Dense(1024, kernel_initializer='RandomUniform'), BatchNormalization(), PReLU(),
    ]:
      nn_list.append(layer(nn_list[-1]))
    return nn_list 
    


# In[ ]:





# In[35]:



print("start running basic neural network")
np.random.seed(1)  # restart random number generator
s1 = Session(parent_dir="/home/alexanderliao/data/Kaggle/competitions/trackml-particle-identification/")
n_events = 100
count = 0
nn_list_basic = MLP(9)

for hits_train, truth_train in s1.get_train_events(n=n_events, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
    count += 1
    print(f"{count}/{n_events}")
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


# In[15]:


os.listdir("/home/alexanderliao/data/Kaggle/competitions/trackml-particle-identification/"+"train/")

