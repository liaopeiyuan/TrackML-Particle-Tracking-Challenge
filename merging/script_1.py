"""
test the clustering score from alex
"""

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from geometric.tools import merge_naive
from utils.session import Session

import lightgbm as lgb

params_1 = {'objective': 'binary', 'boosting': 'gbdt', 'learning_rate': 0.05, 'num_leaves': 32,
            'min_data_in_leaf': 30, 'min_sum_hessian_in_leaf': 40, 'bagging_fraction': 0.8, 'bagging_freq': 1,
            'feature_fraction': 0.7, 'max_delta_step': 0.0, 'lambda_l1': 0.05, 'lambda_l2': 0.05,
            'min_gain_to_split': 0.01, 'scale_pos_weight': 1, 'drop_rate': 0.02, 'verbose': 0,
            'seed': 0}


if __name__ == '__main__':
    np.random.seed(0)
    n_events = 20
    s1 = Session(parent_dir="E:/TrackMLData/")
    for x in s1.get_train_events(n=10, content=[s1.HITS, s1.TRUTH], randomness=True):
        print(x)
    print("bye")


