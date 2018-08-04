"""
test the clustering score from alex
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.cluster import DBSCAN
from geometric.tools import merge_naive
from utils.session import Session
from merging.helix_cluster import run_helix_cluster, clusterer_gen_1, dfh_gen_1


"""
from merging.smart_merge import get_bc_data
cluster_pred_0 = pd.DataFrame({f"step_{i}": val for i, val in enumerate(cluster_pred_0)})
cluster_pred_1 = pd.DataFrame({f"step_{i}": val for i, val in enumerate(cluster_pred_1)})
cluster_pred_2 = pd.DataFrame({f"step_{i}": val for i, val in enumerate(cluster_pred_2)})
cluster_pred_0 = [cluster_pred_0.iloc[:, i].values for i in range(450)]
cluster_pred_1 = [cluster_pred_1.iloc[:, i].values for i in range(450)]
cluster_pred_2 = [cluster_pred_2.iloc[:, i].values for i in range(450)]


w0 = get_pair_weight(temp_data[0][2].values)
w1 = get_pair_weight(temp_data[1][2].values)
w2 = get_pair_weight(temp_data[2][2].values)
y0 = get_flat_adjacency_vector(reassign_noise(temp_data[0][1], temp_data[0][1] == 0)).astype(bool)
y1 = get_flat_adjacency_vector(reassign_noise(temp_data[1][1], temp_data[1][1] == 0)).astype(bool)
y2 = get_flat_adjacency_vector(reassign_noise(temp_data[2][1], temp_data[2][1] == 0)).astype(bool)



x0, y0, w0 = get_bc_data(cluster_pred_0, temp_data[0][1], temp_data[0][2].values, binary_feature=False)
x1, y1, w1 = get_bc_data(cluster_pred_1, temp_data[1][1], temp_data[1][2].values, binary_feature=False)
x2, y2, w2 = get_bc_data(cluster_pred_2, temp_data[2][1], temp_data[2][2].values, binary_feature=False)



"""


if __name__ == '__main__':
    """
    np.random.seed(0)
    n_events = 20
    s1 = Session(parent_dir="E:/TrackMLData/")
    for x in s1.get_train_events(n=10, content=[s1.HITS, s1.TRUTH], randomness=True):
        print(x)
    print("bye")
    """
    
    # TODO: if you have a pair_merge function and a list cluster_pred of cluster ids, you can just call:
    # TODO: reduce(pair_merge, cluster_pred)
s1 = Session("data/")
c = [1.5, 1.5, 0.73, 0.17, 0.027, 0.027]
temp_data = [{"cluster_pred": run_helix_cluster(
    dfh_gen_1(hits, coef=c, n_steps=225, mm=1, stepii=4e-6, z_step=0.5),
    clusterer_gen_1(db_step=5, n_steps=225, adaptive_eps_coef=1, eps=0.0048, min_samples=1, metric="euclidean", p=2, n_jobs=1), parallel=True),
    "truth": truth}
    for hits, truth in s1.get_train_events(n=3, content=[s1.HITS, s1.TRUTH], randomness=True)[1]]


for event_data in temp_data:
    for i, cluster_id in enumerate(event_data["cluster_pred"]):
        event_data["cluster_pred"][i] = reassign_noise(cluster_id, cluster_id == -1)


def get_nn_data(n_events=5):
    ret = []
    count = 0
    for hits, truth in s1.get_train_events(n=n_events, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        count += 1
        print(f"get_nn_data progress: {count}/{n_events}")
        cluster_pred = run_helix_cluster(
            dfh_gen_1(hits, coef=c, n_steps=225, mm=1, stepii=4e-6, z_step=0.5),
            clusterer_gen_1(db_step=5, n_steps=225, adaptive_eps_coef=1, eps=0.0048, min_samples=1, metric="euclidean",
                            p=2, n_jobs=1), parallel=True)
        idx, x, y, w = get_bc_data(cluster_pred, truth["particle_id"], truth["weight"], binary_feature=False, parallel=True)
        ret.append({"idx": idx, "x": x, "y": y, "w": w, "truth": truth})
    return ret


temp_data[0]["cluster_pred"]
mask = temp_data[0]["truth"]["weight"] > 0
raw_weight = temp_data[0]["truth"].loc[mask, "weight"]
raw_particle_id = temp_data[0]["truth"].loc[mask, "particle_id"]

from geometric.tools import reassign_noise



# [max(pd.Series(cluster_id).value_counts().max() for cluster_id in step["cluster_pred"]) for step in temp_data]
# max(_)

# [pd.Series(cluster_id).value_counts().max() for cluster_id in temp_data[0]["cluster_pred"]]

# len(temp_data[0]["cluster_pred"][0])
# len(temp_data[0]["truth"])
# len(temp_data[0]["truth"])

        
# consider one training set first
temp_data[0]

from merging.smart_merge import get_bc_data, get_pair_weight
temp_data = temp_data[:3]
idx0, x0, y0, w0 = get_bc_data(temp_data[0]["cluster_pred"], temp_data[0]["truth"]["particle_id"], temp_data[0]["truth"]["weight"], binary_feature=False, parallel=True)
idx1, x1, y1, w1 = get_bc_data(temp_data[1]["cluster_pred"], temp_data[1]["truth"]["particle_id"], temp_data[1]["truth"]["weight"], binary_feature=False, parallel=True)

# try logistic regression
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, PReLU, Dropout
from merging.nn_tools import f1_metric

nn_1 = Sequential([
    Dense(512, input_shape=(900,)),
    PReLU(), Dropout(0.5), BatchNormalization(),
    
    Dense(256),
    PReLU(), Dropout(0.5), BatchNormalization(),
    
    Dense(256),
    PReLU(), Dropout(0.5), BatchNormalization(),
    
    Dense(128),
    PReLU(), Dropout(0.5), BatchNormalization(),
    
    Dense(1),
    Activation('sigmoid'),
])
nn_1.compile(optimizer='adam', loss='binary_crossentropy')
nn_1.fit(x0, y0, sample_weight=w0, batch_size=2048, epochs=5, validation_data=(x1, y1, w1), shuffle=True, callbacks=[f1_metric])


nn_1.fit(x0, y0, sample_weight=np.ones(x0.shape[0]), batch_size=2048, epochs=5, validation_data=(x1, y1, w1), shuffle=True, callbacks=[f1_metric])


from merging.smart_merge import adjacency_pv_to_cluster_id

import timeit
timeit.timeit("lr_1.predict(x1, batch_size=512)", number=3, globals=globals())
pred_0 = nn_1.predict(x0, batch_size=2048)
pred_1 = nn_1.predict(x1, batch_size=2048)


from geometric.tools import easy_score, easy_sub

for eps in np.arange(0.1, 1.0, 0.1):
    print(eps, easy_score(temp_data[0]["truth"], adjacency_pv_to_cluster_id(temp_data[0]["truth"].shape[0], idx0, pred_0, eps)))
    print(eps, easy_score(temp_data[1]["truth"], adjacency_pv_to_cluster_id(temp_data[1]["truth"].shape[0], idx1, pred_1, eps)))
    
easy_score(temp_data[0]["truth"], adjacency_pv_to_cluster_id(temp_data[0]["truth"].shape[0], idx0, pred_0, 0.25))
easy_score(temp_data[1]["truth"], adjacency_pv_to_cluster_id(temp_data[1]["truth"].shape[0], idx1, pred_1, 0.25))

cluster_0 = adjacency_pv_to_cluster_id(temp_data[0]["truth"].shape[0], idx0, pred_0, 0.25)
easy_score(temp_data[0]["truth"], cluster_0)
easy_score(temp_data[0]["truth"], reassign_noise(cluster_0, cluster_0 == 0))
from merging.alex_hough_8_1 import analyze_truth_perspective
analyze_truth_perspective(temp_data[0]["truth"], easy_sub(temp_data[0]["truth"], cluster_0))



from functools import reduce

pred_0_b = reduce(merge_naive, temp_data[0]["cluster_pred"])
easy_score(temp_data[0]["truth"], pred_0_b)

# from sklearn.linear_model import SGDClassifier
# sgdc1 = SGDClassifier(loss="hinge", penalty="l2", alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=500, tol=1e-3, shuffle=True, verbose=1, epsilon=0.1, n_jobs=-1, random_state=None, learning_rate="optimal", eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False, n_iter=None)
# sgdc1.fit(x0.astype(bool), y0, sample_weight=w0)
# sgdc2 = SGDClassifier(loss="hinge", penalty="l2", alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=500, tol=1e-3, shuffle=True, verbose=1, epsilon=0.1, n_jobs=-1, random_state=None, learning_rate="optimal", eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False, n_iter=None)
# sgdc2.fit(x0, y0, sample_weight=w0)


# x1, y1, w1 = get_bc_data(temp_data[1]["cluster_pred"], temp_data[1]["truth"]["particle_id"], temp_data[1]["truth"]["weight"], binary_feature=False)
from sklearn import metrics
metrics.f1_score(y_true=y1, y_pred=sgdc1.predict(x1.astype(bool)), sample_weight=w1)
metrics.f1_score(y_true=y1, y_pred=sgdc1.predict(x1.astype(bool)))
metrics.f1_score(y_true=y1, y_pred=sgdc2.predict(x1), sample_weight=w1)
metrics.f1_score(y_true=y1, y_pred=sgdc2.predict(x1))

# from sklearn.linear_model import LogisticRegression
# lr1 = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=125, solver='saga', max_iter=100, verbose=1, warm_start=False, n_jobs=-1)
# x1, y1, w1 = get_bc_data(temp_data[1]["cluster_pred"], temp_data[1]["truth"]["particle_id"], temp_data[1]["truth"]["weight"], binary_feature=False)

import lightgbm as lgb
d0 = lgb.Dataset(x0, y0)

params_1 = {'objective': 'binary', 'boosting': 'gbdt', 'learning_rate': 0.05, 'num_leaves': 32,
            'min_data_in_leaf': 20, 'min_sum_hessian_in_leaf': 0.0, 'bagging_fraction': 0.5, 'bagging_freq': 1,
            'feature_fraction': 0.7, 'max_delta_step': 0.0, 'lambda_l1': 0.05, 'lambda_l2': 0.05,
            'min_gain_to_split': 0.01, 'scale_pos_weight': 1, 'drop_rate': 0.02, 'verbose': 0,
            'metric': "f1",
            'seed': 0}
record_dict_1 = {}
m1 = lgb.train(params_1, d0, num_boost_round=50000, valid_sets=[d0, d1], valid_names=["d0", "d1"], early_stopping_rounds=500, evals_result=record_dict_1, verbose_eval=10)


