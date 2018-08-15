"""
start using miaonet_3
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import numpy as np
import pandas as pd

from utils.session import Session
from ete_nn.miaonet_3 import get_nn_data, get_nn_model
from ete_nn.miaonet_1 import train_nn

from trackml.score import score_event

from keras.models import Model
from keras.layers import Input, Embedding, Dense, PReLU, BatchNormalization


def easy_score(truth, pred):
    return score_event(
        truth=truth,
        submission=pd.DataFrame({"hit_id": truth.hit_id, "track_id": pred})
    )


s1 = Session("../data/")

for hits, cells, truth in s1.get_train_events(n=10, content=[s1.HITS, s1.CELLS, s1.TRUTH], randomness=True)[1]:
    di, do, dw = get_nn_data(hits, cells, truth)
    break


mi_1, mo_1 = get_nn_model()

train_nn(mi_1, mo_1, di, do, fw=dw, epochs=100, batch_size=2048, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"], verbose=1)


final_output_layer = Dense(np.max(do) + 1, activation="softmax", trainable=True)(mo_1)

def train_nn(input_layer, output_layer, fx, fy, fw, epochs=10, batch_size=64, loss="categorical_crossentropy", metrics=None, verbose=1):
    final_output_layer = Dense(np.max(fy) + 1, activation="softmax", trainable=True)(output_layer)
    temp_model = Model(inputs=input_layer, outputs=final_output_layer)
    temp_model.compile(optimizer="adam", loss=loss, metrics=metrics)
    temp_model.fit(fx, fy, sample_weight=fw, epochs=epochs, batch_size=batch_size, verbose=verbose)
