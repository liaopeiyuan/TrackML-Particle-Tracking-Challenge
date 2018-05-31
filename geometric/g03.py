"""
g03.py
current best solution
"""

from itertools import product

import numpy as np
import pandas as pd

from sklearn.cluster import dbscan
from sklearn.preprocessing import scale

from geometric.utils import merge_naive, merge_discreet
from geometric.session import Session
from trackml.score import score_event


def sionkowski_43(df, verbose=False):
    df = df.copy()
    df.loc[:, "rt"] = np.sqrt(df.x ** 2 + df.y ** 2)
    df.loc[:, "a0"] = np.arctan2(df.y, df.x)
    df.loc[:, "z1"] = df.z / df.rt
    df.loc[:, "x2"] = df.rt / df.z  # = 1 / df.z1
    dz0 = -7e-4
    stepdz = 1e-5
    stepeps = 5e-6
    mm = 1

    pred = None

    for i in range(101):
        mm *= -1
        dz = mm * (dz0 + i * stepdz)
        df.loc[:, "a1"] = df.a0 + dz * np.abs(df.z)
        df.loc[:, "sina1"] = np.sin(df.a1)
        df.loc[:, "cosa1"] = np.cos(df.a1)
        df.loc[:, "x1"] = df.a1 / df.z1
        dfs = scale(df.loc[:, ["sina1", "cosa1", "z1", "x1", "x2"]])
        dfs *= np.array([1, 1, 0.75, 0.5, 0.5])
        res = dbscan(X=dfs, eps=0.0035 + i * stepeps, min_samples=1, n_jobs=-1, metric="minkowski", p=2)[1]

        pred = merge_naive(pred, res, cutoff=20)

        if verbose:
            official_score = score_event(
                truth=df,
                submission=pd.DataFrame({"hit_id": df.hit_id, "track_id": pred})
            )
            print("score: {:.6f}".format(official_score))
    return pred


def subroutine_1(df):
    sionkowski_43(df, verbose=True)
    return 0


def sionkowski_search(df, feature_weight, p_minkowski=2):
    df = df.copy()
    df.loc[:, "rt"] = np.sqrt(df.x ** 2 + df.y ** 2)
    df.loc[:, "a0"] = np.arctan2(df.y, df.x)
    df.loc[:, "z1"] = df.z / df.rt
    df.loc[:, "x2"] = df.rt / df.z  # = 1 / df.z1
    dz0 = -7e-4
    stepdz = 1e-5
    stepeps = 5e-6
    mm = 1

    score_list = []
    pred = None

    for i in range(150):
        mm *= -1
        dz = mm * (dz0 + i * stepdz)
        df.loc[:, "a1"] = df.a0 + dz * np.abs(df.z)
        df.loc[:, "sina1"] = np.sin(df.a1)
        df.loc[:, "cosa1"] = np.cos(df.a1)
        df.loc[:, "x1"] = df.a1 / df.z1
        dfs = scale(df.loc[:, ["sina1", "cosa1", "z1", "x1", "x2"]])
        dfs *= feature_weight
        res = dbscan(X=dfs, eps=0.0035 + i * stepeps, min_samples=1, n_jobs=-1, metric="minkowski", p=p_minkowski)[1]

        pred = merge_naive(pred, res, cutoff=20)

        official_score = score_event(
            truth=df,
            submission=pd.DataFrame({"hit_id": df.hit_id, "track_id": pred})
        )
        score_list.append(official_score)
    return score_list


if __name__ == "__main__":
    print("start running script g03.py")
    s1 = Session(parent_dir="E:/TrackMLData/")

    result_record = pd.DataFrame(columns=['w1', 'w2', 'w3', 'w4', 'w5', 'best_n', 'best_score'])

    n_events = 30

    i = 0

    feature_weight_list = product(np.arange(0.05, 1 + 1e-12, 0.05), repeat=5)

    for feature_weight in feature_weight_list:
        result_record.loc[i, ['w1', 'w2', 'w3', 'w4', 'w5']] = feature_weight
        score_per_step = 0
        for hits, truth in s1.get_train_events(n=n_events, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
            score_per_step += np.array(
                sionkowski_search(hits.merge(truth, how="left", on="hit_id"), feature_weight, p_minkowski=2)
            )
        score_per_step /= n_events
        result_record.loc[i, 'best_n'] = np.argmax(score_per_step)
        result_record.loc[i, 'best_score'] = np.max(score_per_step)
        i += 1
        if i % 1000 == 0:
            print(result_record.loc[result_record.best_score > result_record.best_score.quantile(0.98), ["best_n", "best_score"]])

    for hits, truth in s1.remove_train_events(n=10, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        print("=" * 120)
        hits = hits.merge(truth, how="left", on="hit_id")
        subroutine_1(hits)

