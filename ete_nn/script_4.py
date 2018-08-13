import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import numpy as np
import pandas as pd

from ete_nn.miaonet_1 import get_basic_nn, train_nn, prepare_df, get_target, permute_target, augment_1, get_feature
from utils.session import Session

from trackml.score import score_event

from keras.models import Model
from keras.layers import Input, Embedding, Dense, PReLU, BatchNormalization


# np.random.seed(1)  # restart random number generator
s1 = Session("../data/")
nn_list_basic = get_basic_nn(3)
for hits, truth in s1.get_train_events(n=10, content=[s1.HITS, s1.TRUTH], randomness=True)[1]: break

df = prepare_df(hits, truth)
fx = get_feature(augment_1(df, np.random.rand()*2*np.pi))
fy = get_target(df)

train_nn(nn_list_basic, fx, fy, basic_trainable=False, epochs=5, batch_size=1024, loss="sparse_categorical_crossentropy", verbose=1)
