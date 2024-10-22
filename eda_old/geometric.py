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

from eda_old.arsenal import HITS, TRUTH
from eda_old.arsenal import get_directories, get_event_name


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


def merge_2_clusters(pred_1, pred_2, cutoff=21):
    # pred_1, pred_2 # hit id -> track id
    c1, c2 = Counter(pred_1), Counter(pred_2)  # track id -> track size
    n1 = np.array([c1[c_id] for c_id in pred_1])  # hit id -> track size
    n2 = np.array([c2[c_id] for c_id in pred_2])  # hit id -> track size
    pred = pred_1.copy()
    pred[(n2 > n1) & (n2 < cutoff)] = max(pred_1) + 1 + pred_2[(n2 > n1) & (n2 < cutoff)]
    pred[(pred_1 == -1) & (pred_2 != -1)] = max(pred_1) + 1 + pred_2[(pred_1 == -1) & (pred_2 != -1)]
    return pred


def merge_cluster(pred_list):
    # pred_list = pred_list[::-1]  # reverse the list
    pred = pred_list[0]
    for next_pred in pred_list[1:]:
        pred = merge_2_clusters(pred, next_pred)
    return pred


def predict_multiple_cluster(xyz_array, n_theta=20):
    pred_list = []
    df = pd.DataFrame()
    for theta in np.linspace(0.0, 180.0, n_theta):
        df["hx"], df["hy"], df["hz"] = helix_1(*helix_2(xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2], theta))
        pred = dbscan(X=scale(df), eps=0.007, min_samples=1, n_jobs=-1)[1]
        pred_list.append(pred)
    return pred_list


def run_multiple_cluster(xyz_array, truth, n_theta=20):
    pred_list = []
    df = pd.DataFrame()  # the length always matches
    for theta in np.linspace(0.0, 180.0, n_theta):
        # print("*" * 60)
        print("theta={:.4f}".format(theta), end="; ")
        df["hx"], df["hy"], df["hz"] = helix_1(*helix_2(xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2], theta))

        pred = dbscan(X=scale(df), eps=0.007, min_samples=1, n_jobs=-1)[1]

        # print("-1 entries: {}/{}".format(sum(pred == -1), pred.size))
        # c1 = Counter(pred)  # cluster id -> cluster size
        # pred[pred == c_id] = -1 for c_id in c1 if c1[c_id] > 30 # this is not an expression
        # c2 = Counter(n for c_id, n in c1.most_common())  # cluster size -> number of clusters with that size

        # print(sorted(c2.most_common()))

        print("score={:.6f}".format(score_event(truth=truth, submission=pd.DataFrame({"hit_id": truth.hit_id, "track_id": pred}))))

        pred_list.append(pred)

        # print("running dbscan test")
        # test_dbscan(eps_list=eps_list, hit_id=truth.hit_id, data=df, truth=truth, scaling=True)
    return pred_list


def merge2_func1(pred_1, pred_2):
    c1, c2 = Counter(pred_1), Counter(pred_2)  # track id -> track size
    n1 = np.array([c1[c_id] for c_id in pred_1])  # hit id -> track size
    n2 = np.array([c2[c_id] for c_id in pred_2])  # hit id -> track size
    pred = pred_1.copy()
    pred[(n2 > n1)] = max(pred_1) + 1 + pred_2[(n2 > n1)]
    return pred


def merge2_func2(pred_1, pred_2, cutoff=21):
    c1, c2 = Counter(pred_1), Counter(pred_2)  # track id -> track size
    n1 = np.array([c1[c_id] for c_id in pred_1])  # hit id -> track size
    n2 = np.array([c2[c_id] for c_id in pred_2])  # hit id -> track size
    pred = pred_1.copy()
    pred[(n2 > n1) & (n2 < cutoff)] = max(pred_1) + 1 + pred_2[(n2 > n1) & (n2 < cutoff)]
    pred[(pred_1 == -1) & (pred_2 != -1)] = max(pred_1) + 1 + pred_2[(pred_1 == -1) & (pred_2 != -1)]
    return pred


def aggregate_helix_1(xyz: np.ndarray, truth=None):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x*x + y*y + z*z)  # radius in spherical coordinate system
    rt = np.sqrt(x*x + y*y)  # radius in cylindrical coordinate system
    a0 = np.arctan2(y, x)  # angle in cylindrical coordinate system
    z1 = z / rt

    pred = None
    merge2_func = merge2_func1

    dz = -0.00012
    stepdz = 0.00001
    for ii in range(13):
        dz += ii * stepdz
        a1 = a0 + dz * z * np.sign(z)
        x1 = a1 / z1
        x2 = 1 / z1
        x3 = x1 + x2
        dfs = scale(np.column_stack([a1, z1, x1, x2, x3]))
        res = dbscan(X=dfs, eps=.0035, min_samples=1, n_jobs=-1)[1]

        # todo: test code, remove when done ============================================================================
        if False and truth is not None:
            temp_var = lambda df: sum(np.var(df, axis=0))
            temp_c = Counter(res)  # cluster id -> cluster size
            temp_d1 = []  # cluster id -> cluster size and cluster variance in hidden space
            for temp_c_id in temp_c:
                if temp_c[temp_c_id] > 3:
                    temp_d1.append((temp_c[temp_c_id], temp_var(dfs[res == temp_c_id, :])))
            temp_t = Counter(truth.particle_id)  # track/particle id -> cluster size
            temp_d2 = []  # track id -> track size and track variance in hidden space
            for temp_t_id in temp_t:
                if temp_t_id != 0 and temp_t[temp_t_id] > 3:
                    temp_d2.append((temp_t[temp_t_id], temp_var(dfs[truth.particle_id == temp_t_id, :])))
            pd.DataFrame(np.array(temp_d1)).to_csv("cluster_variance.csv", index=False, header=False)
            pd.DataFrame(np.array(temp_d2)).to_csv("track_variance.csv", index=False, header=False)
            exit("early exit on purpose")

        # todo: test code, remove when done ============================================================================

        if ii == 0:
            pred = res
        else:
            pred = merge2_func(pred, res)
        if truth is not None:
            print("current score: ", score_event(truth=truth, submission=pd.DataFrame({"hit_id": truth.hit_id, "track_id": pred})))

    dz = 0.00012
    stepdz = -0.00001
    for ii in range(13):
        dz += ii * stepdz
        a1 = a0 + dz * z * np.sign(z)
        x1 = a1 / z1
        x2 = 1 / z1
        x3 = x1 + x2
        dfs = scale(np.column_stack([a1, z1, x1, x2, x3]))
        res = dbscan(X=dfs, eps=.0035, min_samples=1, n_jobs=-1)[1]
        pred = merge2_func(pred, res)
        if truth is not None:
            print("current score: ", score_event(truth=truth, submission=pd.DataFrame({"hit_id": truth.hit_id, "track_id": pred})))
    return pred


print("start running the script")
TRAIN_DIR, TEST_DIR, DETECTORS_DIR, SAMPLE_SUBMISSION_DIR, TRAIN_EVENT_ID_LIST, TEST_EVENT_ID_LIST = get_directories()

n_event = 40  # TODO:important parameter
n_train = 20  # TODO:important parameter
event_id_list = np.random.choice(TRAIN_EVENT_ID_LIST, size=n_event, replace=False)
train_id_list = event_id_list[:n_train]  # training set
val_id_list = event_id_list[n_train:]  # validation set

for event_id in train_id_list:
    print('='*120)
    hits, truth = load_event(TRAIN_DIR + get_event_name(event_id), [HITS, TRUTH])
    # pred_list = run_multiple_cluster(hits[["x", "y", "z"]].values, truth=truth, n_theta=40)
    pred = aggregate_helix_1(hits[["x", "y", "z"]].values, truth=truth)

    print("final score: ", end="")
    print(score_event(truth=truth, submission=pd.DataFrame({
        "hit_id": truth.hit_id,
        "track_id": pred,
    })))


if False:
    sub_list = []
    for event_id in TEST_EVENT_ID_LIST:
        print(event_id)
        hits, = load_event(TEST_DIR + get_event_name(event_id), [HITS])
        pred_list = predict_multiple_cluster(hits[["x", "y", "z"]].values, n_theta=40)
        sub = pd.DataFrame({"hit_id": hits.hit_id, "track_id": merge_cluster(pred_list)})
        sub.insert(0, "event_id", event_id)
        sub_list.append(sub)
    final_submission = pd.concat(sub_list)
    final_submission.to_csv("sub0001.csv", sep=",", header=True, index=False)

