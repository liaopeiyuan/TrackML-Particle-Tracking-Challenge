import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import numpy as np
import pandas as pd

from ete_nn.miaonet_1 import get_basic_nn, train_nn, prepare_df, get_target, permute_target, augment_1, get_feature, get_feature_cylindrical
from utils.session import Session

from trackml.score import score_event

from keras.models import Model
from keras.layers import Input, Embedding, Dense, PReLU, BatchNormalization


def easy_score(truth, pred):
    return score_event(
        truth=truth,
        submission=pd.DataFrame({"hit_id": truth.hit_id, "track_id": pred})
    )


from ete_nn.miaonet_2 import InceptionResNetV2

# np.random.seed(1)  # restart random number generator
s1 = Session("../data/")

for hits, cells, truth in s1.get_train_events(n=10, content=[s1.HITS, s1.CELLS, s1.TRUTH], randomness=True)[1]:
    break

max(hits.module_id.max() for hits, in s1.get_test_event(n=10000, content=[s1.HITS], randomness=True)[1])

temp_set_1 = set()
for cells, in s1.get_train_events(n=10000, content=[s1.CELLS], randomness=True)[1]:
    temp_set_1.add((cells.ch0.nunique(), cells.ch1.nunique(), cells.ch0.min(), cells.ch1.min(), cells.ch0.max(), cells.ch1.max()))
    



df = prepare_df(hits, truth)
# fx = get_feature(augment_1(df, np.random.rand()*2*np.pi))
# fx = get_feature(augment_1(df, np.random.rand()*2*np.pi))
fx = get_feature_cylindrical(df)
fy = get_target(df)

fw = df["weight"] * (df.shape[0] / df["weight"].sum())

# fw = df["weight"].clip(0.0, 5.0)

nn_input, nn_output = InceptionResNetV2(input_shape=(3,))

# nn_list_basic = get_basic_nn(3)
train_nn(nn_input, nn_output, fx, fy, fw=fw.values, epochs=2000, batch_size=2048, loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"], verbose=1)
#

# final_encoder = Model(inputs=nn_list_basic[0], outputs=nn_list_basic[-1])

# e1 = final_encoder.predict(get_feature(hits[["x", "y", "z"]]), batch_size=2048)
# from sklearn.cluster import DBSCAN
# c1 = DBSCAN(eps=0.001, min_samples=1, n_jobs=-1).fit_predict(e1)
# easy_score(truth, c1)

