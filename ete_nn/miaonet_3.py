import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Activation, GlobalMaxPool1D


def get_nn_data(hits_df: pd.DataFrame, cells_df: pd.DataFrame, truth_df: pd.DataFrame=None,
                use_volume=True, use_layer=True, use_module=True):
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
        
    x = {
        "input_geometric": hits_df[["x", "y", "z"]].values
    }  # input dictionary to neural network
    
    # preprocess dataframe values
    if use_volume:
        volume_dict = {7: 0, 8: 1, 9: 2, 12: 3, 13: 4, 14: 5, 16: 6, 17: 7, 18: 8}
        hits_df["volume_id"] = hits_df["volume_id"].map(volume_dict)
        x["input_volume"] = hits_df[["volume_id"]].values
    
    if use_layer:
        layer_dict = {2: 0, 4: 1, 6: 2, 8: 3, 10: 4, 12: 5, 14: 6}
        hits_df["layer_id"] = hits_df["layer_id"].map(layer_dict)
        x["input_layer"] = hits_df[["layer_id"]].values
        
    if use_module:
        hits_df["module_id"] -= 1
        x["input_module"] = hits_df[["module_id"]].values
    
    if truth_df is None:
        return x, None, None
    else:
        do_id = truth_df["particle_id"].values
        do_id = LabelEncoder().fit_transform(do_id)
        dw = truth_df["weight"].values
        dw = dw * (dw.shape[0] / dw.sum())
        return x, do_id, dw
    
    
def get_nn_model(geometric_size=3, use_volume=3, use_layer=3, use_module=32, use_ch0=16, use_ch1=16, use_value=True):
    """
    geometric_size: the size of the geometric input (e.g. cartesian coordinates: x, y, z -> size=3)
    use_volume: size of volume embedding
    """
    input_geometric = Input(shape=(geometric_size,), name="input_geometric")
    
    input_list = [input_geometric]
    concat_list = [input_geometric]
    
    if use_volume > 0:
        embed_dim_in_volume = 9
        embed_dim_out_volume = use_volume
        input_volume = Input(shape=(1,), name="input_volume")
        input_list.append(input_volume)
        embed_volume = Embedding(input_dim=embed_dim_in_volume, output_dim=embed_dim_out_volume, name="embed_volume")(input_volume)
        flat_volume = Flatten(name="flat_volume")(embed_volume)
        concat_list.append(flat_volume)
        
    if use_layer > 0:
        embed_dim_in_layer = 7
        embed_dim_out_layer = use_layer
        input_layer = Input(shape=(1,), name="input_layer")
        input_list.append(input_layer)
        embed_layer = Embedding(input_dim=embed_dim_in_layer, output_dim=embed_dim_out_layer, name="embed_layer")(input_layer)
        flat_layer = Flatten(name="flat_layer")(embed_layer)
        concat_list.append(flat_layer)
        
    if use_module > 0:
        embed_dim_in_module = 3192
        embed_dim_out_module = use_module
        input_module = Input(shape=(1,), name="input_module")
        input_list.append(input_module)
        embed_module = Embedding(input_dim=embed_dim_in_module, output_dim=embed_dim_out_module, name="embed_module")(input_module)
        flat_module = Flatten(name="flat_module")(embed_module)
        concat_list.append(flat_module)
    
    if any((use_ch0, use_ch1, use_value)):
        cells_input_list, cells_output = get_cells_model(use_ch0, use_ch1, use_value)
        input_list.extend(cells_input_list)
        concat_list.append(cells_output)
        
    x = Concatenate(name="concat_hits")(concat_list) if len(concat_list) > 1 else input_geometric
    
    for i in range(10):
        x = Dense(units=128, use_bias=False)(x)
        x = BatchNormalization(scale=False)(x)
        x = Activation("relu")(x)
    return input_list, x


def get_cells_model(use_ch0=16, use_ch1=16, use_value=True):
    input_list = []
    concat_list = []
    if use_ch0 > 0:
        embed_dim_in_ch0, embed_dim_out_ch0 = 1200, use_ch0
        input_ch0 = Input(shape=(None,), name="input_ch0")
        input_list.append(input_ch0)
        embed_ch0 = Embedding(input_dim=embed_dim_in_ch0, output_dim=embed_dim_out_ch0, name="embed_ch0")(input_ch0)
        concat_list.append(embed_ch0)
    if use_ch1 > 0:
        embed_dim_in_ch1, embed_dim_out_ch1 = 1280, use_ch1
        input_ch1 = Input(shape=(None,), name="input_ch1")
        input_list.append(input_ch1)
        embed_ch1 = Embedding(input_dim=embed_dim_in_ch1, output_dim=embed_dim_out_ch1, name="embed_ch1")(input_ch1)
        concat_list.append(embed_ch1)
    if use_value:
        input_value = Input(shape=(None, 1), name="input_value")
        input_list.append(input_value)
        concat_list.append(input_value)
    # at least one feature (ch0, ch1, value) should be used in cells, otherwise, stop at upper-level function
    x = Concatenate(axis=-1, name="concat_cells")(concat_list) if len(concat_list) > 1 else concat_list[0]
    for i in range(5):
        x = Dense(units=64, use_bias=False)(x)
        x = BatchNormalization(scale=False)(x)
        x = Activation("relu")(x)
    x = GlobalMaxPool1D()(x)
    return input_list, x


# def fast_cells_groupby(cells_df: pd.DataFrame, col):
#     idx = (np.where(cells_df.iloc[:-1, "hit_id"] != cells_df.iloc[:, -1])[0] + 1).tolist()
    
    
def get_cells_data(cells_df: pd.DataFrame):
    cells_gb = cells_df.groupby("hit_id")
    ch0 = cells_gb["ch0"].apply(np.array)
    ch1 = cells_gb["ch1"].apply(np.array)
    value = cells_gb["value"].apply(lambda x: np.array(x).reshape(-1, 1))
    return {"input_ch0": ch0, "input_ch1": ch1, "input_value": value}
    
    from timeit import timeit
    n = 20
    timeit('cells_gb["ch0"].apply(np.array), cells_gb["ch1"].apply(np.array), cells_gb["value"].apply(np.array)', number=n, globals=globals()) / n

# temp_m.fit((lambda d: {x: d[x].tolist() for x in d})(ci_d), y, batch_size=2048, verbose=1)


def get_all_data(hits_df: pd.DataFrame, cells_df: pd.DataFrame, truth_df: pd.DataFrame=None,
                 use_volume=True, use_layer=True, use_module=True, use_ch0=True, use_ch1=True, use_value=True):
    hits_df = hits_df.copy()
    hits_df.index = hits_df.hit_id.values
    cells_df = cells_df.copy()
    cells_df.index = cells_df.hit_id.values
    
    if truth_df is not None:
        truth_df = truth_df.copy()
        truth_df.index = truth_df.hit_id.values
        # training, drop unimportant hits
        drop_idx = truth_df.index[((truth_df.groupby("particle_id")["particle_id"].transform("count") <= 3) | (truth_df["weight"] == 0)).values]
        hits_df.drop(drop_idx, axis=0, inplace=True)
        truth_df.drop(drop_idx, axis=0, inplace=True)
        cells_df.drop(drop_idx, axis=0, inplace=True)
        # prepare label encoder
        le_1 = LabelEncoder()
        le_1.fit(truth_df["particle_id"].values)
        # normalize weight
        truth_df["weight"] = truth_df["weight"] * (truth_df.shape[0] / truth_df["weight"].sum())
        
    # preprocess dataframe values
    if use_volume:
        volume_dict = {7: 0, 8: 1, 9: 2, 12: 3, 13: 4, 14: 5, 16: 6, 17: 7, 18: 8}
        hits_df["volume_id"] = hits_df["volume_id"].map(volume_dict)
        
    if use_layer:
        layer_dict = {2: 0, 4: 1, 6: 2, 8: 3, 10: 4, 12: 5, 14: 6}
        hits_df["layer_id"] = hits_df["layer_id"].map(layer_dict)
        
    if use_module:
        hits_df["module_id"] -= 1
        
    # divide into batches by the number of cells
    hit_id_to_n_cells = cells_df.groupby("hit_id").size().rename("n_cells")
    n_cells_to_hit_id = hit_id_to_n_cells.groupby(hit_id_to_n_cells).apply(lambda x: x.index).to_dict()
    
    ret_list = []
    for n_cells in n_cells_to_hit_id:
        hit_idx = n_cells_to_hit_id[n_cells]
        x = {
            "input_geometric": hits_df.loc[hit_idx, ["x", "y", "z"]].values
        }  # input dictionary to neural network
        
        if use_volume:
            x["input_volume"] = hits_df.loc[hit_idx, "volume_id"].values.reshape((-1, 1))
        if use_layer:
            x["input_layer"] = hits_df.loc[hit_idx, "layer_id"].values.reshape((-1, 1))
        if use_module:
            x["input_module"] = hits_df.loc[hit_idx, "module_id"].values.reshape((-1, 1))
        if use_ch0:
            x["input_ch0"] = cells_df.loc[hit_idx, "ch0"].values.reshape((-1, n_cells))
        if use_ch1:
            x["input_ch1"] = cells_df.loc[hit_idx, "ch1"].values.reshape((-1, n_cells))
        if use_value:
            x["input_value"] = cells_df.loc[hit_idx, "value"].values.reshape((-1, n_cells, 1))
            
        if truth_df is None:
            ret_list.append(x)
        else:
            y = le_1.transform(truth_df.loc[hit_idx, "particle_id"].values)
            w = truth_df.loc[hit_idx, "weight"].values
            ret_list.append((x, y, w))
            
    if truth_df is None:
        return ret_list, None
    else:
        return ret_list, le_1.classes_.shape[0]

    
# train neural network and returns a final accuracy score

def train_nn_all(inputs, outputs, data_list, n_classes, outer_epochs=10, inner_epochs=4, batch_size=2048, loss="sparse_categorical_crossentropy", metrics=None, verbose=1):
    final_output_layer = Dense(n_classes, activation="softmax", trainable=True)(outputs)
    temp_model = Model(inputs=inputs, outputs=final_output_layer)
    temp_model.compile(optimizer="adam", loss=loss, metrics=metrics)
    for i in range(outer_epochs):  # number of epochs for the entire dataset
        print(f"outer epoch: {i+1}/{outer_epochs}")
        for x, y, w in data_list:
            # inner_epochs is the number of epochs for each n_cells batch
            temp_model.fit(x, y, sample_weight=w, epochs=inner_epochs, batch_size=batch_size, verbose=verbose)
        scores = temp_model.evaluate_generator(iter(data_list), steps=len(data_list))
        print(f"loss={scores[0]}, accuracy={scores[1]}")
    return temp_model.evaluate_generator(iter(data_list), steps=len(data_list))

            
def train_nn(inputs, outputs, fx, fy, fw, epochs=10, batch_size=64, loss="categorical_crossentropy", metrics=None, verbose=1):
    final_output_layer = Dense(np.max(fy) + 1, activation="softmax", trainable=True)(outputs)
    temp_model = Model(inputs=inputs, outputs=final_output_layer)
    temp_model.compile(optimizer="adam", loss=loss, metrics=metrics)
    temp_model.fit(fx, fy, sample_weight=fw, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return log_loss(y_true=fy, y_pred=temp_model.predict(fx, batch_size=batch_size), sample_weight=fw)

