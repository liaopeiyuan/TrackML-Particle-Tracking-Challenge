import numpy as np
import pandas as pd

from utils.session import Session

# from keras.models import Model
# from keras.layers import Input, Embedding


def prepare_df(hits_df, truth_df):
    df = truth_df[["hit_id", "particle_id"]].merge(hits_df[["hit_id", "x", "y", "z", "volume_id", "layer_id", "module_id", "weight"]], on="hit_id")
    # drop useless rows
    df = df.merge(pd.DataFrame(hits.groupby("particle_id").size().rename("track_size")), left_on="particle_id", right_index=True)
    df = df.loc[(df["track_size"] > 3) & (df["particle_id"] != 0), ["x", "y", "z", "volume_id", "layer_id", "module_id", "weight"]]
    # prepare categorical variables for embedding
    volume_dict = {7: 0, 8: 1, 9: 2, 12: 3, 13: 4, 14: 5, 16: 6, 17: 7, 18: 8}
    layer_dict = {2: 0, 4: 1, 6: 2, 8: 3, 10: 4, 12: 5, 14: 6}
    df["volume_id"] = df["volume_id"].map(volume_dict)
    df["layer_id"] = df["layer_id"].map(layer_dict)
    df["module_id"] -= 1
    return df

    
    
    
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
    
# print(hits.columns.tolist())
# print(truth.columns.tolist())

len(set(range(1, 3193)).difference(module_set))

min(module_set)
max(module_set)