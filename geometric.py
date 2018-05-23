"""
geometric.py

use geometric transformations to solve the problem

by Tianyi Miao
"""
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.cluster import dbscan
from sklearn.preprocessing import scale

from trackml.dataset import load_event
from trackml.score import score_event

from arsenal import get_directories, get_event_name, StaticFeatureEngineer
from arsenal import HITS, CELLS, PARTICLES, TRUTH


def test_dbscan(eps_list, hit_id, data, truth, scaling):
    for eps in eps_list:
        pred = pd.DataFrame({
            "hit_id": hit_id,
            "track_id": dbscan(
                X=(scale(data) if scaling else data),
                eps=eps, min_samples=1, n_jobs=-1
            )[1]  # return labels, not including core samples
        })
        print("eps={}, score:  ".format(eps), end='\t')
        print(score_event(truth=truth, submission=pred))


def helix_1(x, y, z):
    """
    helix unrolling that works since the beginning
    """
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    return x/d, y/d, z/r


def helix_2(x, y, z, theta=20, v=1000):
    """
    rotate the helix by an angle of 20
    theta: angle of rotation
    v: normalization constant
    """
    d = np.sqrt(x ** 2 + y ** 2)
    # r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = (theta / 180) * np.pi * (d / v)
    hx = x * np.cos(-phi) - y * np.sin(-phi)
    hy = x * np.sin(-phi) + y * np.cos(-phi)
    return hx, hy, z


def merge_cluster(pred_list):
    pred = pred_list[0]
    for next_pred in pred_list[1:]:
        next_pred[next_pred != -1] += max(pred) + 1
        pred[pred == -1] = next_pred[pred == -1]
    return pred


def run_multiple_cluster(xyz_array, truth, eps_list=(0.004, 0.007, 0.008, 0.01, 0.02), n_theta=20):
    df = pd.DataFrame()
    pred_list = []
    for theta in np.linspace(0.0, 180.0, n_theta):
        # print("*" * 60)
        print("theta = ", theta)
        df["hx"], df["hy"], df["hz"] = helix_1(*helix_2(xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2], theta))

        pred = dbscan(X=scale(df), eps=0.008, min_samples=3, n_jobs=-1)[1]

        print("-1 entries: {}/{}".format(sum(pred == -1), pred.size))
        c1 = Counter(pred)  # cluster id -> cluster size
        c2 = Counter(n for c_id, n in c1.most_common())  # cluster size -> number of clusters with that size

        # print(sorted(c2.most_common()))

        print("current score: ", end="")
        print(score_event(truth=truth, submission=pd.DataFrame({"hit_id": truth.hit_id, "track_id": pred})))

        pred_list.append(pred)

        # print("running dbscan test")
        # test_dbscan(eps_list=eps_list, hit_id=truth.hit_id, data=df, truth=truth, scaling=True)
    return pred_list

print("start running the script")
TRAIN_DIR, TEST_DIR, DETECTORS_DIR, SAMPLE_SUBMISSION_DIR, TRAIN_EVENT_ID_LIST, TEST_EVENT_ID_LIST = get_directories()

n_event = 40  # TODO:important parameter
n_train = 20  # TODO:important parameter
event_id_list = np.random.choice(TRAIN_EVENT_ID_LIST, size=n_event, replace=False)
train_id_list = event_id_list[:n_train]  # training set
val_id_list = event_id_list[n_train:]  # validation set

for event_id in train_id_list:
    print('='*120)
    truth, = load_event(TRAIN_DIR + get_event_name(event_id), [TRUTH])
    pred_list = run_multiple_cluster(truth[["tx", "ty", "tz"]].values, truth=truth, n_theta=20)

    print("final score: ", end="")
    print(score_event(truth=truth, submission=pd.DataFrame({
        "hit_id": truth.hit_id,
        "track_id": merge_cluster(pred_list) + 1
    })))



