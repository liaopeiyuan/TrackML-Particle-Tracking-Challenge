import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, dok_matrix, lil_matrix, hstack as sparse_hstack, csr_matrix
import networkx as nx

from itertools import combinations
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from trackml.score import score_event
from trackml.dataset import load_event
from functools import reduce
from tqdm import tqdm


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
            event_ids = self._train_event_id_list[:n]
            self._train_event_id_list = self._train_event_id_list[n:] + self._train_event_id_list[:n]
        
        event_names = [Session.get_event_name(event_id) for event_id in event_ids]
        return event_names, \
               (load_event(self._parent_dir + self._train_dir + event_name, content) for event_name in event_names)
    
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
        return event_names, \
               (load_event(self._parent_dir + self._train_dir + event_name, content) for event_name in event_names)
    
    def get_test_event(self, n=3, content=(HITS, TRUTH), randomness=True):
        n = min(n, len(self._test_event_id_list))
        if randomness:
            event_ids = np.random.choice(self._test_event_id_list, size=n, replace=False).tolist()
        else:
            event_ids, = self._test_event_id_list[:n]
            self._test_event_id_list = self._test_event_id_list[n:] + self._test_event_id_list[:n]
        
        event_names = [Session.get_event_name(event_id) for event_id in event_ids]
        return event_names, \
               (load_event(self._parent_dir + self._test_dir + event_name, content) for event_name in event_names)
    
    def remove_test_events(self, n=10, content=(HITS, CELLS), randomness=False):
        n = min(n, len(self._test_event_id_list))
        if randomness:
            event_ids = np.random.choice(self._test_event_id_list, size=n, replace=False).tolist()
            for event_id in event_ids:
                self._test_event_id_list.remove(event_id)
        else:
            event_ids, self._test_event_id_list = self._test_event_id_list[:n], self._test_event_id_list[n:]
        event_names = [Session.get_event_name(event_id) for event_id in event_ids]
        return event_names, \
               (load_event(self._parent_dir + self._test_dir + event_name, content) for event_name in event_names)
    
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


# ======================================================================================================================
def easy_score(truth, pred):
    return score_event(
        truth=truth,
        submission=pd.DataFrame({"hit_id": truth.hit_id, "track_id": pred})
    )


def easy_sub(truth, pred):
    return pd.DataFrame({"hit_id": truth.hit_id, "track_id": pred})


def label_encode(y):
    return LabelEncoder().fit_transform(y)


def reassign_noise(labels: np.ndarray, mask):
    """
    assign noisy points (labeled with key_value such as -1 or 0) to their own clusters of size 1
    """
    ret = labels.copy()
    ret[mask] = np.arange(np.sum(mask)) + np.max(ret) + 1
    return ret


def merge_naive(pred_1, pred_2, cutoff=20):
    if pred_1 is None:
        return pred_2
    d = pd.DataFrame(data={'s1': pred_1, 's2': pred_2})
    d['N1'] = d.groupby('s1')['s1'].transform('count')
    d['N2'] = d.groupby('s2')['s2'].transform('count')
    max_s1 = d['s1'].max() + 1
    cond = np.where((d['N2'].values > d['N1'].values) & (d['N2'].values < cutoff))
    s1 = d['s1'].values
    s1[cond] = d['s2'].values[cond] + max_s1
    return label_encode(s1)


# ======================================================================================================================
def pred_wrapper(arg):
    return arg[1].fit_predict(arg[0])


def run_helix_cluster(dfh_gen, clusterer_gen, parallel=True):
    if parallel:
        with mp.Pool(24) as pool_1:
            ret = list(pool_1.map(pred_wrapper, zip(dfh_gen, clusterer_gen)))
            pool_1.close()
            return ret
    else:
        return map(pred_wrapper, zip(dfh_gen, clusterer_gen))


# ======================================================================================================================
def get_flat_adjacency_vector(cluster_id):
    n = cluster_id.shape[0]
    ret = dok_matrix((n * (n - 1) // 2, 1), dtype=np.uint8)
    pred = pd.DataFrame({"cluster_id": cluster_id})
    pred = pred.join(pred["cluster_id"].value_counts().rename("cluster_size"), on="cluster_id")  # get cluster size
    pred["sample_index"] = pred.index
    pred = pred.loc[pred["cluster_size"] > 1, :]  # eliminate singletons to save groupby time
    
    def subroutine(sub_df):
        cluster_size = sub_df["cluster_size"].iloc[0]  # cluster size is the same for all points in a cluster
        for j, i in combinations(sub_df["sample_index"], r=2):  # j < i is guaranteed
            ret[i * (i - 1) // 2 + j, 0] = cluster_size
            # if adjacent, return positive cluster size, which provides more information than binary indicator
            # could be useful for machine learning algorithms
    
    pred.groupby("cluster_id").agg(subroutine)  # use agg instead of apply to avoid running the first group twice
    return ret


def get_pair_weight(idx, weight):
    # idx is a 2d array: n_samples * 2
    weight = np.array(weight)
    ret_w = weight[idx[:, 0]] + weight[idx[:, 1]]
    ret_w = ret_w * (len(ret_w) / np.sum(ret_w))
    return ret_w


def get_bc_data(cluster_pred, particle_id=None, weight=None, binary_feature=False, parallel=True):
    """
    :param cluster_pred: list (len = n_steps) of cluster id prediction arrays (length n_samples)
    :param particle_id: (n_samples,)
    :param weight: (n_samples,)
    :param binary_feature: use binary/bool as adjacency feature, rather than cluster size
    :return:
    prepare for binary classification
    """
    n = len(cluster_pred[0])
    print("Preparing ret_x")
    with mp.Pool() as pool_1:
        ret_x = sparse_hstack(blocks=(
            pool_1.map(get_flat_adjacency_vector, cluster_pred) if parallel else
            map(get_flat_adjacency_vector, cluster_pred)
        ),
            format="csr", dtype=(bool if binary_feature else np.uint8))
    print("Preparing mask")
    # if True in mask, the two points are assigned to the same cluster in at least one step
    mask = ret_x.indptr[1:] != ret_x.indptr[:-1]  # if True in mask, will be kept in memory
    ret_x = ret_x[mask, :]
    print("Preparing idx")
    ret_idx = np.array(np.tril_indices(n, -1)).T[mask]
    print("Preparing ret_w")
    ret_w = None if weight is None else get_pair_weight(idx=ret_idx, weight=weight)
    print("Preparing ret_y")
    ret_y = None if particle_id is None else \
    get_flat_adjacency_vector(reassign_noise(particle_id, particle_id == 0)).astype(bool)[mask].toarray().ravel()
    # notice: noisy hits (particle_id == 0) will be reassigned to facilitate track size computation
    # then, use a classifier such as lr/lgb/nn to fit (ret_x, ret_y, ret_w), ret_w is optional
    print("Done")
    return ret_idx, ret_x.astype(np.float32), ret_y, ret_w


def adjacency_pv_to_cluster_id(n, idx, apv, eps=0.5):
    """
    :param apv: predicted adjacency probability vector from binary classifier (only the ones in mask)
    :param eps: threshold to consider two points adjacent
    :param mask: boolean mask with shape n*(n-1)//2
    :return:
    """
    # n = int((apv.shape[0] * 2) ** 0.5) + 1  # the shape of the symmetric matrix
    # this is the inverse formula from n*(n-1)//2
    apv = np.array(apv).ravel()
    g1 = nx.from_edgelist(idx[apv > eps].tolist())
    c_id = 1
    ret = np.zeros(n)
    for component in nx.connected_components(g1):
        ret[list(component)] = c_id
        c_id += 1
    return ret


def predict_bc(cluster_pred, predict_func, eps=0.5, binary_feature=False, parallel=True):
    # predict_func must return a probability vector
    idx, x, _, _ = get_bc_data(cluster_pred, binary_feature=binary_feature, parallel=parallel)
    ret = adjacency_pv_to_cluster_id(cluster_pred.shape[0], idx, predict_func(x), eps)
    return ret


# ======================================================================================================================
'''
import keras
from sklearn.metrics import precision_recall_fscore_support


class F1Callback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0], batch_size=2048))).round()
        val_target = self.validation_data[1]
        val_weight = self.validation_data[2]  # notice: the validation data is weighted
        _val_precision, _val_recall, _val_fscore, _val_support = precision_recall_fscore_support(y_true=val_target, y_pred=val_predict, sample_weight=val_weight, average="binary")
        self.val_f1s.append(_val_fscore)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(f"val_fscore: {_val_fscore}, val_recall: {_val_recall}, val_precision: {_val_precision}")
        # print “ — val_f1: % f — val_precision: % f — val_recall % f” % (_val_f1, _val_precision, _val_recall)
        return

# ======================================================================================================================
f1_metric = F1Callback()
s1 = Session("data/")
c = [1.5, 1.5, 0.73, 0.17, 0.027, 0.027]


def get_nn_data(ret, n_events=5):
    count = 0
    for hits, truth in s1.get_train_events(n=n_events, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        count += 1
        print(f"get_nn_data progress: {count}/{n_events}")
        cluster_pred = run_helix_cluster(
            dfh_gen_1(hits, coef=c, n_steps=225, mm=1, stepii=4e-6, z_step=0.5),
            clusterer_gen_1(db_step=5, n_steps=225, adaptive_eps_coef=1, eps=0.0048, min_samples=1, metric="euclidean",
                            p=2, n_jobs=1), parallel=True)
        for i, cluster_id in enumerate(cluster_pred):
            if any(cluster_id == -1):
                cluster_pred[i] = reassign_noise(cluster_id, cluster_id == -1)
        idx, x, y, w = get_bc_data(cluster_pred, truth["particle_id"], truth["weight"], binary_feature=False, parallel=True)
        ret.append({"idx": idx, "x": x, "y": y, "w": w, "truth": truth})


data_list = []
get_nn_data(data_list, 5)


def train_nn(nn_model, data, valid_dict, epochs, batch_size=2048, callbacks=None):
    if callbacks is None:
        callbacks = [f1_metric]
    for i in range(epochs):
        print(f"Large Epoch: {i}/{epochs}")
        for train_dict in data:
            nn_model.fit(train_dict["x"], train_dict["y"], sample_weight=train_dict["w"], epochs=1, shuffle=True,
                         validation_data=(valid_dict["x"], valid_dict["y"], valid_dict["w"]), batch_size=batch_size,
                         callbacks=callbacks)



from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, PReLU, Dropout, Input, Reshape, Conv1D, MaxPool1D

nn_1 = Sequential([
    Dense(512, input_shape=(900,)),
    PReLU(), Dropout(0.3), BatchNormalization(),
    Dense(256), PReLU(), Dropout(0.3), BatchNormalization(),
    Dense(256), PReLU(), Dropout(0.3), BatchNormalization(),
    Dense(128), PReLU(), Dropout(0.3), BatchNormalization(),
    Dense(128), PReLU(), Dropout(0.3), BatchNormalization(),
    Dense(128), PReLU(), Dropout(0.3), BatchNormalization(),
    Dense(1, activation="sigmoid"),
])
nn_1.compile(optimizer='adam', loss='binary_crossentropy')
train_nn(nn_1, data_list[:-1], data_list[-1], epochs=20, batch_size=2048, callbacks=[f1_metric])

nn_1.fit(data_list[0]["x"], data_list[0]["y"], sample_weight=data_list[0]["w"], epochs=20, shuffle=True,
         validation_data=(data_list[-1]["x"], data_list[-1]["y"], data_list[-1]["w"]), batch_size=2048,
         callbacks=[f1_metric])
pred_0 = nn_1.predict(data_list[0]["x"], batch_size=2048)
for eps in np.arange(0.1, 1.0, 0.1):
    print(f"{eps:.4}", easy_score(data_list[0]["truth"], adjacency_pv_to_cluster_id(data_list[0]["truth"].shape[0], data_list[0]["idx"], pred_0, eps)))

# perfect cluster from perfect y
cluster_p = adjacency_pv_to_cluster_id(data_list[0]["truth"].shape[0], data_list[0]["idx"], data_list[0]["y"], 0.5)

# test truly perfect scenario
pid_temp = data_list[0]["truth"]["particle_id"].values
pid_temp = reassign_noise(pid_temp, pid_temp == 0)
idx0, x0, y0, w0 = get_bc_data([pid_temp, pid_temp], data_list[0]["truth"]["particle_id"], data_list[0]["truth"]["weight"], parallel=True)
easy_score(data_list[0]["truth"], adjacency_pv_to_cluster_id(data_list[0]["truth"].shape[0], idx0, y0, 0.5))

for eps in np.arange(0.1, 1.0, 0.1):
    print(f"{eps:.4}", easy_score(data_list[0]["truth"], adjacency_pv_to_cluster_id(data_list[0]["truth"].shape[0], data_list[0]["idx"], data_list[0]["y"], eps)))

def get_nn_model(input_shape):
    nn_list = [
        Input(input_shape, sparse=True),
    ]
    for layer in [
        Reshape((1, -1)),  # prepare for 1d convolution
        
        Conv1D(filters=128, kernel_size=4, strides=1, padding="valid", kernel_regularizer=None, use_bias=False),
        BatchNormalization(),
        PReLU(),
        Conv1D(filters=128, kernel_size=4, strides=1, padding="valid", kernel_regularizer=None, use_bias=False),
        BatchNormalization(),
        PReLU(),
        Conv1D(filters=128, kernel_size=4, strides=1, padding="valid", kernel_regularizer=None, use_bias=False),
        BatchNormalization(),
        PReLU(),
        MaxPool1D(pool_size=2, padding="valid"),
    
        Conv1D(filters=64, kernel_size=4, strides=1, padding="valid", kernel_regularizer=None, use_bias=False),
        BatchNormalization(),
        PReLU(),
        Conv1D(filters=64, kernel_size=4, strides=1, padding="valid", kernel_regularizer=None, use_bias=False),
        BatchNormalization(),
        PReLU(),
        Conv1D(filters=64, kernel_size=4, strides=1, padding="valid", kernel_regularizer=None, use_bias=False),
        BatchNormalization(),
        PReLU(),
        MaxPool1D(pool_size=2, padding="valid"),
        
    ]:
        nn_list.append(layer(nn_list[-1]))
    return nn_list
    
'''
# import lightgbm as lgb
# from timeit import timeit
# temp_mask = np.random.rand(49995000) > 0.9
# num = 10
# timeit("np.array(np.tril_indices(10000, -1)).T[temp_mask]", number=num, globals=globals()) / num
# timeit("(lambda x: np.array([x[0][temp_mask], x[1][temp_mask]]).T)(np.tril_indices(10000, -1))", number=num, globals=globals()) / num

'''
def score_upper_bound(dfh_gen, cluster_gen, truth, parallel=True):
    # upper bound is monotonic with: from all true pairs, how many true pairs can be clustered in at least one step?
    # however, a pathological solution: group everything into the same cluster -> highest upper bound
    print("Preparing cluster_pred")
    cluster_pred = run_helix_cluster(dfh_gen, cluster_gen, parallel=parallel)
    n = len(cluster_pred[0])
    print("Preparing ret_x")
    with mp.Pool() as pool_1:
        ret_x = sparse_hstack(blocks=(
            pool_1.map(get_flat_adjacency_vector, cluster_pred) if parallel else
            map(get_flat_adjacency_vector, cluster_pred)
        ),
        format="csr", dtype=bool)
    print("Preparing mask")
    # if True in mask, the two points are assigned to the same cluster in at least one step
    mask = ret_x.indptr[1:] != ret_x.indptr[:-1]  # if True in mask, will be kept in memory
    del ret_x
    print("Preparing idx")
    ret_idx = np.array(np.tril_indices(n, -1)).T[mask]
    print("Preparing ret_y")
    particle_id = truth["particle_id"].values
    ret_y = get_flat_adjacency_vector(reassign_noise(particle_id, particle_id == 0)).astype(bool)[mask].toarray().ravel()
    print("calculating score upper bound")
    ub_score = easy_score(truth, adjacency_pv_to_cluster_id(n, ret_idx, ret_y, eps=0.5))
    print(f"score upper bound: {ub_score}")
    
    print("calculating benchmark score")
    bm_pred = reduce(merge_naive, cluster_pred)
    bm_score = easy_score(truth, bm_pred)
    print(f"benchmark score: {bm_score}\n(if this is significantly lower, the clustering algorithm is giving too many false positives)")
    return ub_score, bm_score
'''


def dfh_gen_1(df, coef, n_steps=225, mm=1, stepii=4e-6, z_step=0.5):
    for z0 in np.arange(-5.5, 5.5, z_step):
        df['z'] = df['z'] + z0  # TODO: the r later may be different
        df['r'] = np.sqrt(df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2)
        df['rt'] = np.sqrt(df['x'] ** 2 + df['y'] ** 2)
        df['a0'] = np.arctan2(df['y'], df['x'])
        df['zdivrt'] = df['z'] / df['rt']
        df['zdivr'] = df['z'] / df['r']
        df['xdivr'] = df['x'] / df['r']
        df['ydivr'] = df['y'] / df['r']
        for ii in np.arange(0, n_steps * stepii, stepii):
            for jj in range(2):
                mm = mm * (-1)
                df['a1'] = df['a0'].values - np.nan_to_num(np.arccos(mm * ii * df['rt'].values))
                df['sina1'] = np.sin(df['a1'])
                df['cosa1'] = np.cos(df['a1'])
                ss = StandardScaler()
                dfs = ss.fit_transform(df[['sina1', 'cosa1', 'zdivrt', 'zdivr', 'xdivr', 'ydivr']].values)
                # dfs = scale_ignore_nan(dfh[['sina1','cosa1','zdivrt','zdivr','xdivr','ydivr']])
                dfs = np.multiply(dfs, coef)
                yield dfs


def cluster_gen_1(db_step=5, n_steps=225, adaptive_eps_coef=1, eps=0.05, min_samples=1, metric="euclidean", p=2, n_jobs=1):
    """
    default code provided by Alex on Slack, August 2nd
    """
    for db in np.arange(min_samples, 10, db_step):
        for ii in range(1, n_steps + 1):
            for jj in range(2):
                eps_new = eps + ii * adaptive_eps_coef * 1e-5
                yield DBSCAN(eps=eps_new, min_samples=db, n_jobs=n_jobs, metric=metric, metric_params=None, p=p)


# new generation
def dfh_gen_2(df, w_xy_cluster, w_xy_r, n_step, size_step=4e-6):
    # total steps: n_astep * 2
    # w_xy_cluster: weight on x/y during clustering, 0 to 1.5
    # w_xy_cluster: weight on x/y during radius->angle rotation, 0 to 1.5
    for jj in range(1):
        # df = df[['x', 'y', 'z']].copy()
        # df['z'] = df['z']  # TODO: the r later may be different
        df['a0'] = np.arctan2(df['y'], df['x'])
        # df['r'] = np.sqrt(df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2)
        df['rt'] = np.sqrt(df['x'] ** 2 + df['y'] ** 2)
        df['zdivrt'] = df['z'] / df['rt']
        df['a_delta'] = np.sqrt(df['x']**2*w_xy_r + df['y']**2*w_xy_r + df['z']**2*(3-w_xy_r*2)) * size_step
        for ii in range(n_step):
            for mm in [1, -1]:
                df['a1'] = df['a0'] + df['a_delta'] * mm * ii
                df['sina1'] = np.sin(df['a1'])
                df['cosa1'] = np.cos(df['a1'])
                ss = StandardScaler()
                dfs = ss.fit_transform(df[['sina1', 'cosa1', 'zdivrt']].values)
                dfs = np.multiply(dfs, [w_xy_cluster, w_xy_cluster, 3-w_xy_cluster*2])
                yield dfs

                
def cluster_gen_2(n_step=225, adaptive_eps_coef=1, eps_0=0.05, min_samples=1, metric="euclidean", p=2, n_jobs=1):
    # for db in np.arange(min_samples, 10, db_step):
    for ii in range(1, n_step + 1):
        for jj in range(2):
            eps_new = eps_0 + ii * adaptive_eps_coef * 1e-5
            yield DBSCAN(eps=eps_new, min_samples=min_samples, n_jobs=n_jobs, metric=metric, metric_params=None, p=p)




# exploring possible features
s1 = Session("data/")
np.random.seed(679433641)
for hits, truth in s1.get_train_events(n=1, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
    break

'''
cluster_pred = run_helix_cluster(
    dfh_gen_1(hits, coef=[1.5, 1.5, 0.73, 0.17, 0.027, 0.027], n_steps=225, mm=1, stepii=4e-6, z_step=0.5),
    cluster_gen_1(db_step=5, n_steps=225, adaptive_eps_coef=1, eps=0.0048, min_samples=1, metric="euclidean", p=2,
                  n_jobs=1),
    True
)
cluster_pred = run_helix_cluster(
    dfh_gen_2(hits, coef=[1, 1, 1], n_step=225, size_step=4e-6),
    cluster_gen_2(n_step=225, adaptive_eps_coef=1, eps_0=0.03),
    parallel=True)
easy_score(truth, reduce(merge_naive, cluster_pred))
'''

from bayes_opt import BayesianOptimization

'''
def calc_helix_score(n_step, w_xy_cluster, w_xy_r, size_step, log2_eps, log2_adaptive_eps_coef):
    adaptive_eps_coef = 2 ** log2_adaptive_eps_coef
    n_step = int(n_step)
    size_step = float(size_step)
    eps_0 = 2 ** log2_eps
    cluster_pred = run_helix_cluster(
        dfh_gen_2(hits, w_xy_cluster=w_xy_cluster, w_xy_r=w_xy_r, n_step=n_step, size_step=size_step),
        cluster_gen_2(n_step=n_step, adaptive_eps_coef=adaptive_eps_coef, eps_0=eps_0),
        parallel=True)
    return easy_score(truth, reduce(merge_naive, cluster_pred))

cluster_pred = run_helix_cluster(
        dfh_gen_2(hits, coef=[1, 1, 1], n_step=200, size_step=4e-6),
        cluster_gen_2(n_step=225, adaptive_eps_coef=1, eps_0=0.03),
        parallel=True)


helix_bo = BayesianOptimization(f=calc_helix_score, pbounds={
    "n_step": (10, 1000),
    "w_xy_cluster": (0.0, 1.5),
    "w_xy_r": (0.0, 1.5),
    "size_step": (0, 0.1),
    "log2_eps": (-10, -3),
    "log2_adaptive_eps_coef": (-5, 5),
}, random_state=None, verbose=1)

helix_bo.explore({"n_step": [100, 200], "w_xy_cluster": [1.5], "w_xy_r": [1.5], "size_step": [1e-5], })

helix_bo.maximize(init_points=10, n_iter=25)


d1 = list(cluster_gen_1(db_step=5, n_steps=225, adaptive_eps_coef=1, eps=0.0048, min_samples=1, metric="euclidean", p=2, n_jobs=1))
d2 = list(cluster_gen_2(n_step=225, adaptive_eps_coef=1, eps_0=0.0048))

cluster_pred = run_helix_cluster(
    dfh_gen_1(hits, coef=[1.5, 1.5, 0.73, 0.17, 0.027, 0.027], n_steps=225, mm=1, stepii=4e-6, z_step=0.5),
    cluster_gen_2(n_step=225, adaptive_eps_coef=1, eps_0=0.0048),
    True
)
'''


# debugging generator function:
d1 = (lambda x, y, z: np.sqrt(x**2 + y**2 + z**2))(hits.x, hits.y, hits.z)
d2 = (lambda df, w_xy_ra: np.dot(df[["x", "y", "z"]] ** 2, [w_xy_ra, w_xy_ra, 3 - w_xy_ra * 2]))(hits, 1)


def dfh_gen_3(df, step_init=0, n_step=100, w_xy_ra=1.0, w_xy_rz=1.0, w_xy_c=1.0, a_step=1e-5, z_step=0.0, log=None):
    if log is None or not log:
        log = {"a_delta": [], "z_delta": []}
    # will yield n_step * 2 steps
    df = df[["x", "y", "z"]].copy()
    df["z0"] = df["z"]  # TODO: z-shifting
    df["a0"] = np.arctan2(df["y"], df["x"])  # default angle
    df["ra"] = np.sqrt(np.dot(df[["x", "y", "z"]] ** 2, [w_xy_ra, w_xy_ra, 3 - w_xy_ra * 2]))  # weighted radius for angle
    df["rz"] = np.sqrt(np.dot(df[["x", "y", "z"]] ** 2, [w_xy_rz, w_xy_rz, 3 - w_xy_rz * 2]))  # weighted radius for z
    df["rt"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    for ii in range(step_init, step_init + n_step):  # step index
        for mm in (-1, 1):  # direction multiplier
            if ii == 0 and mm == 1:
                break
            a_delta = mm * ii * a_step
            z_delta = mm * ii * z_step
            log["a_delta"].append(a_delta)
            log["z_delta"].append(z_delta)
            df["a_delta"] = a_delta * df["ra"]
            df["z_delta"] = z_delta * df["rz"]
            df["a1"] = df["a0"] + df["a_delta"]
            df["z1"] = df["z0"] + np.sign(df["z0"]) * df["z_delta"]  # z coordinates are symmetric by the xy-plane
            # perparing features for clustering
            df["sin_a1"] = np.sin(df["a1"])
            df["cos_a1"] = np.cos(df["a1"])
            df["z1_div_rt"] = df["z1"] / df["rt"]
            dfs = StandardScaler().fit_transform(df[["sin_a1", "cos_a1", "z1_div_rt"]])
            dfs = np.multiply(dfs, [w_xy_c, w_xy_c, 3 - w_xy_c * 2])
            yield dfs


def cluster_gen_3(step_init=0, n_step=225, eps_init=0.05, eps_step=1e-5, min_samples=1, metric="euclidean", p=2,
                  n_jobs=1, log=None):
    if log is None or not log:
        log = {"eps": []}
    for ii in range(step_init, step_init + n_step):
        eps = eps_init + ii * eps_step
        for mm in (-1, 1):
            if ii == 0 and mm == 1:
                break
            log["eps"].append(eps)
            yield DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs, metric=metric, metric_params=None, p=p)


# a larger step index (ii) indicates further deviation from the original coordinates
log1, log2 = {}, {}
cluster_pred = run_helix_cluster(
    dfh_gen_3(hits, step_init=-50, n_step=120, w_xy_ra=1, w_xy_rz=0.7, w_xy_c=0.9, a_step=1e-5, z_step=1e-5, log=log1),
    cluster_gen_3(step_init=-50, n_step=120, eps_init=4e-3, eps_step=-5e-6, log=log2), parallel=True)
easy_score(truth, reduce(merge_naive, cluster_pred))
easy_score(truth, reduce(merge_naive, cluster_pred[::-1]))

log1_list, log2_list = [{}], [{}]


def calc_helix_score(step_init, n_step, w_xy_ra, w_xy_c, eps_init_e3, eps_step_e6, eps_step_sign):
    step_init = int(step_init)
    n_step = int(n_step)
    eps_init = eps_init_e3 * 1e-3
    eps_step = eps_step_e6 * 1e-6 * np.sign(eps_step_sign)
    try:
        cluster_pred = run_helix_cluster(
            dfh_gen_3(hits, step_init=step_init, n_step=n_step, w_xy_ra=w_xy_ra, w_xy_rz=0, w_xy_c=w_xy_c, a_step=1e-5, z_step=0.0, log=log1_list[-1]),
            cluster_gen_3(step_init=step_init, n_step=n_step, eps_init=eps_init, eps_step=eps_step, log=log2_list[-1]),
            parallel=True)
    except ValueError:
        return -1.0
    log1_list.append({})
    log2_list.append({})
    return easy_score(truth, reduce(merge_naive, cluster_pred))

helix_bo = BayesianOptimization(f=calc_helix_score, pbounds={
    "step_init": (-100, 50),
    "n_step": (120, 500),
    "w_xy_ra": (0.0, 1.5),
    "w_xy_c": (0.0, 1.5),
    "eps_init_e3": (1, 10),
    "eps_step_e6": (1, 10),
    "eps_step_sign": (-1, 1),
}, random_state=None, verbose=1)

helix_bo.maximize(init_points=2000, n_iter=1000, kappa=5)


helix_bo.maximize(init_points=100, n_iter=3000, kappa=15)

easy_score(truth, reduce(merge_naive, cluster_pred[::2]))
easy_score(truth, reduce(merge_naive, cluster_pred[::-2]))
easy_score(truth, reduce(merge_naive, cluster_pred[::3]))
easy_score(truth, reduce(merge_naive, cluster_pred[::-3]))
easy_score(truth, reduce(merge_naive, (cluster_pred[j] for j in np.hstack(np.array([0,3]) + 4 * i for i in range(60))[:-1])))

