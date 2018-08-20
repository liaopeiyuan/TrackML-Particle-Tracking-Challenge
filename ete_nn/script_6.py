"""
try etenn with all inputs, including cells

best parameters from script_5.py:
geometric_size=3, use_volume=4, use_layer=0, use_module=128

initialization in command shell:
cd /rscratch/xuanyu/trackML/Kaggle-TrackML
export PATH=/rscratch/xuanyu/KAIL/anaconda3/bin:$PATH
export PATH="$PATH:/usr/local/cuda-8.0/bin"
export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64"
source activate p36-tf
git pull 

"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Activation, GlobalMaxPool1D, PReLU

from utils.session import Session
from ete_nn.miaonet_4 import get_all_data, get_full_model, train_nn_all

from trackml.score import score_event
from itertools import product


def easy_score(truth, pred, hit_id):
    return score_event(
        truth=truth,
        submission=pd.DataFrame({"hit_id": hit_id, "track_id": pred})
    )


s1 = Session("../data/")

for hits, cells, truth in s1.get_train_events(n=10, content=[s1.HITS, s1.CELLS, s1.TRUTH], randomness=True)[1]:
    break
    
params_1 = {
    'use_volume': 4,
    'use_layer': 0,
    'use_module': 128,
    'use_ch0': 16,
    'use_ch1': 16,
    'use_value': True,
}


def cells_architecture_1(x):
    for i in range(6):
        x = Dense(units=64, use_bias=True, activation="relu")(x)
    return x


def full_architecture_1(x):
    for i in range(10):
        x = Dense(units=128, use_bias=True)(x)
        x = Activation("relu")(x)
    return x


m1_i, m1_o = get_full_model(
    geometric_size=3, full_architecture=full_architecture_1, cells_architecture=cells_architecture_1, **params_1)
data_list, hit_idx, n_classes = get_all_data(hits, cells, truth, **params_1)
train_nn_all(m1_i, m1_o, data_list, n_classes, trainable=True, epochs=2000, steps_per_epoch=80, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"], verbose=1)


'''
m1_i, m1_o = get_nn_model(geometric_size=3, **params_1)
data_list, n_classes = get_all_data(hits, cells, truth, **params_1)
train_nn_all(m1_i, m1_o, data_list, n_classes, epochs=2000, steps_per_epoch=80, batch_size=2048, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"], verbose=1)
'''