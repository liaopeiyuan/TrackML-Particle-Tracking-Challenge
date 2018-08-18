"""
start using miaonet_3
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import numpy as np
import pandas as pd

from utils.session import Session
from ete_nn.miaonet_3 import get_nn_data, get_nn_model, train_nn

from trackml.score import score_event
from itertools import product

from keras.models import Model
from keras.layers import Input, Embedding, Dense, PReLU, BatchNormalization


def easy_score(truth, pred):
    return score_event(
        truth=truth,
        submission=pd.DataFrame({"hit_id": truth.hit_id, "track_id": pred})
    )


s1 = Session("../data/")

for hits, cells, truth in s1.get_train_events(n=10, content=[s1.HITS, s1.CELLS, s1.TRUTH], randomness=True)[1]:
    break


# di["input_geometric"] = cartesian_to_cylindrical(di["input_geometric"])


record_1 = {}
for params in product((False, True), repeat=3):
    di, do, dw = get_nn_data(hits, cells, truth, *params)
    mi_1, mo_1 = get_nn_model(3, *params)
    record_1[params] = train_nn(mi_1, mo_1, di, do, fw=dw, epochs=4500, batch_size=2048, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"], verbose=1)

record_1 = {
    (0, 0, 0): 0.5985390183552075,
    (0, 0, 32): 0.46324630122516786,
    (0, 3, 0): 0.5722868060800796,
    (0, 3, 32): 0.42994079133385765,
    (3, 0, 0): 0.5616555043692665,
    (3, 0, 32): 0.3969729097340809,
    (3, 3, 0): 0.6702962944311427,
    (3, 3, 32): 0.44920655267255877,
    (3, 2, 32): 0.5086306505932866,
    (4, 0, 32): 0.3955510315078606,
    (4, 0, 40): 0.35224301137572755,
    (4, 0, 64): 0.3751855208956262,
    (4, 0, 80): 0.38791492805179917,
    (4, 0, 128): 0.3022749020164693,
    (4, 0, 160): 0.386663249450646,
    (4, 0, 200): 0.3376686399662939,
    (4, 0, 256): 0.33658441136367623
}


record_1 = {}
for params in [(4, 0, 128)]:
    di, do, dw = get_nn_data(hits, cells, truth, *params)
    mi_1, mo_1 = get_nn_model(3, *params)
    record_1[params] = train_nn(mi_1, mo_1, di, do, fw=dw, epochs=10000, batch_size=2048, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"], verbose=1)



def cartesian_to_cylindrical(xyz):
    r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    a = np.arctan2(xyz[:, 1], xyz[:, 0])
    z = xyz[:, 2]
    return np.vstack([r, a, z]).T


def cartesian_to_cylindrical_2(xyz):
    r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    a = np.arctan2(xyz[:, 1], xyz[:, 0])
    phi = np.arctan2(xyz[:, 2], r)
    return np.vstack([r, a, phi]).T


