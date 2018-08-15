import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Activation


def get_nn_data(hits_df: pd.DataFrame, cells_df: pd.DataFrame, truth_df: pd.DataFrame=None):
    # notice that hit_id in hits_df, cells_df, and truth_df is already sorted
    hits_df = hits_df.copy()
    hits_df.index = hits_df.hit_id
    if truth_df is not None:
        truth_df = truth_df.copy()
        truth_df.index = truth_df.hit_id
    
    # create mask for only the important hits
    if truth_df is not None:  # training, drop unimportant hits
        mask = ((truth_df.groupby("particle_id")["particle_id"].transform("count") > 3) & (truth_df["weight"] > 0)).values
        hits_df = hits_df.loc[mask, :]
        truth_df = truth_df.loc[mask, :]
    
    # preprocess dataframe values
    volume_dict = {7: 0, 8: 1, 9: 2, 12: 3, 13: 4, 14: 5, 16: 6, 17: 7, 18: 8}
    layer_dict = {2: 0, 4: 1, 6: 2, 8: 3, 10: 4, 12: 5, 14: 6}
    hits_df["volume_id"] = hits_df["volume_id"].map(volume_dict)
    hits_df["layer_id"] = hits_df["layer_id"].map(layer_dict)
    hits_df["module_id"] -= 1
    
    # data input and data output
    di_geometric = hits_df[["x", "y", "z"]].values
    di_volume = hits_df[["volume_id"]].values
    di_layer = hits_df[["layer_id"]].values
    di_module = hits_df[["module_id"]].values
    # output_tp = truth_df[["tpx", "tpy", "tpz"]]

    x = {
        "input_geometric": di_geometric,
        "input_volume": di_volume,
        "input_layer": di_layer,
        "input_module": di_module,
    }
    
    if truth_df is None:
        return x, None, None
    else:
        do_id = truth_df["particle_id"].values
        do_id = LabelEncoder().fit_transform(do_id)
        dw = truth_df["weight"].values
        dw = dw * (dw.shape[0] / dw.sum())
        return x, do_id, dw
    
    
def get_nn_model():
    embed_dim_in_ch0 = 1200
    embed_dim_in_ch1 = 1280
    embed_dim_out_ch0 = 16
    embed_dim_out_ch1 = 16
    
    embed_dim_in_volume = 9
    embed_dim_in_layer = 7
    embed_dim_in_module = 3192
    
    embed_dim_out_volume = 3
    embed_dim_out_layer = 3
    embed_dim_out_module = 32
    
    input_geometric = Input(shape=(3,), name="input_geometric")
    input_volume = Input(shape=(1,), name="input_volume")
    input_layer = Input(shape=(1,), name="input_layer")
    input_module = Input(shape=(1,), name="input_module")
    # add embedding layers
    embed_volume = Embedding(input_dim=embed_dim_in_volume, output_dim=embed_dim_out_volume, name="embed_volume")(input_volume)
    embed_layer = Embedding(input_dim=embed_dim_in_layer, output_dim=embed_dim_out_layer, name="embed_layer")(input_layer)
    embed_module = Embedding(input_dim=embed_dim_in_module, output_dim=embed_dim_out_module, name="embed_module")(input_module)
    
    flat_volume = Flatten(name="flat_volume")(embed_volume)
    flat_layer = Flatten(name="flat_layer")(embed_layer)
    flat_module = Flatten(name="flat_module")(embed_module)
    
    x = Concatenate(name="concat_hits")([input_geometric, flat_volume, flat_layer, flat_module])
    for i in range(10):
        x = Dense(units=128, use_bias=False)(x)
        x = BatchNormalization(scale=False)(x)
        x = Activation("relu")(x)
    return [input_geometric, input_volume, input_layer, input_module], x



