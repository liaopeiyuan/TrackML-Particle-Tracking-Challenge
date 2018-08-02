"""
test the clustering score from alex
"""

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from geometric.tools import merge_naive
from utils.session import Session




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



import lightgbm as lgb
d0 = lgb.Dataset(x0, y0, weight=w0)
d1 = lgb.Dataset(x1, y1, weight=w1)
d2 = lgb.Dataset(x2, y2, weight=w2)
params_1 = {'objective': 'binary', 'boosting': 'gbdt', 'learning_rate': 0.05, 'num_leaves': 32,
            'min_data_in_leaf': 20, 'min_sum_hessian_in_leaf': 0.0, 'bagging_fraction': 0.8, 'bagging_freq': 1,
            'feature_fraction': 0.7, 'max_delta_step': 0.0, 'lambda_l1': 0.05, 'lambda_l2': 0.05,
            'min_gain_to_split': 0.01, 'scale_pos_weight': 1, 'drop_rate': 0.02, 'verbose': 0,
            'metric':
            'seed': 0}
lgb.train(params_1, d0, num_boost_round=10000, valid_sets=[d0, d1], valid_names=["d0", "d1"], )




if __name__ == '__main__':
    np.random.seed(0)
    n_events = 20
    s1 = Session(parent_dir="E:/TrackMLData/")
    for x in s1.get_train_events(n=10, content=[s1.HITS, s1.TRUTH], randomness=True):
        print(x)
    print("bye")


