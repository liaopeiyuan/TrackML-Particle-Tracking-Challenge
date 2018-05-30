"""
g01.py

first geometric script

by Tianyi Miao

some notes:
official_score = score_event(truth=truth, submission=pd.DataFrame({"hit_id": truth.hit_id, "track_id": pred}))
hcv_score = homogeneity_completeness_v_measure(labels_true=reassign_noise(truth.particle_id, idx=(truth.particle_id == 0)), labels_pred=pred)

"""

from collections import Counter

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.preprocessing import scale, LabelEncoder
from sklearn.cluster import dbscan
from sklearn.metrics import homogeneity_completeness_v_measure

from geometric.session import Session
from trackml.score import score_event


def label_encode(y):
    return LabelEncoder().fit_transform(y)


def helix_1(x, y, z):
    """
    helix unrolling that works since the beginning
    """
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    return x/d, y/d, z/r


def reassign_noise(labels: np.ndarray, idx):
    """
    assign noisy points (labeled with key_value such as -1 or 0) to their own clusters of size 1
    """
    ret = labels.copy()
    ret[idx] = np.array(list(range(int(np.sum(idx))))) + np.max(ret) + 1
    return ret


def merge_two_1(pred_1, pred_2, cutoff=20):
    """
    naive cluster merging:
    iterate over hits; if a hit belongs to a larger cluster in pred_2, it is reassigned
    """
    c1, c2 = Counter(pred_1), Counter(pred_2)  # track id -> track size
    n1, n2 = np.vectorize(c1.__getitem__)(pred_1), np.vectorize(c2.__getitem__)(pred_2)  # hit id -> track size
    pred = pred_1.copy()
    idx = (n2 > n1) & (n2 < cutoff)
    pred[idx] = max(pred_1) + 1 + pred_2[idx]
    return label_encode(pred)


def merge_two_2(pred_1, pred_2, cutoff=21):
    """
    discreet cluster merging (less likely to reassign points)
    iterate over clusters in pred_2; np.sum(n1[idx]) < c2[track]**2 -> pred[idx] = d + track
    this is self-documenting
    """
    c1, c2 = Counter(pred_1), Counter(pred_2)  # track id -> track size
    n1, n2 = np.vectorize(c1.__getitem__)(pred_1), np.vectorize(c2.__getitem__)(pred_2)  # hit id -> track size
    pred = reassign_noise(pred_1, n1 > cutoff)
    pred_2 = reassign_noise(pred_2, n2 > cutoff)
    n1[n1 > cutoff] = 1
    n2[n2 > cutoff] = 1
    d = max(pred) + 1
    for track in c2:
        if c2[track] < 3:
            continue
        idx = pred_2 == track
        if np.sum(n1[idx]) < c2[track]**2:
            pred[idx] = d + track
    return label_encode(pred)


def aggregate_helix_2(x, y, z, truth=None, merge_func=merge_two_2):
    rc = np.sqrt(x * x + y * y)  # radius in xy plane
    ac = np.arctan2(y, x)  # angle in xy plane


def aggregate_helix_1(x, y, z, truth=None, merge_func=merge_two_1):
    # if truth is not None:
    #     new_particle_id = reassign_noise(truth.particle_id, idx=(truth.particle_id == 0))
    # r = np.sqrt(x * x + y * y + z * z)
    rt = np.sqrt(x * x + y * y)
    a0 = np.arctan2(y, x)
    z1 = z / rt
    dz0 = -0.00070
    stepdz = 0.00001
    stepeps = 0.000005
    mm = 1

    pred = None

    for i in range(200):
        mm *= -1
        dz = mm * (dz0 + i * stepdz)
        a1 = a0 + dz * np.abs(z)
        x1 = a1 / z1
        x2 = 1 / z1
        x3 = x1 + x2
        dfs = scale(np.column_stack((a1, z1, x1, x2, x3)))
        res = dbscan(X=dfs, eps=0.0035 + max(i, 90) * stepeps, min_samples=1, n_jobs=-1)[1]

        if pred is None:
            pred = res
        else:
            pred = merge_func(pred, res, cutoff=20)
        if truth is not None:
            official_score = score_event(
                truth=truth,
                submission=pd.DataFrame({"hit_id": truth.hit_id, "track_id": pred})
            )
            print("official score: {:.6f}".format(official_score))
            """
            hcv_score = homogeneity_completeness_v_measure(
                labels_true=new_particle_id,
                labels_pred=pred)
            print("homogeneity:    {:.6f}\n"
                  "completeness:   {:.6f}\n"
                  "v measure:      {:.6f}".format(*hcv_score))
            """
    return pred


def subroutine_1(hits, truth):
    print("=" * 120)

    x, y, z = hits.x, hits.y, hits.z
    hidden = scale(np.column_stack(helix_1(x, y, z)))

    for eps in (0.001, 0.006, 0.008, 0.01, 0.04):
        print("-" * 120)
        print("eps={}".format(eps))
        pred = dbscan(X=hidden, eps=eps, min_samples=1, n_jobs=1)[1]

        official_score = score_event(truth=truth, submission=pd.DataFrame({"hit_id": truth.hit_id, "track_id": pred}))
        print("official score: {:.6f}".format(official_score))

        hcv_score = homogeneity_completeness_v_measure(
            labels_true=truth.particle_id,
            labels_pred=pred
        )
        print("homogeneity:    {:.6f}\n"
              "completeness:   {:.6f}\n"
              "v measure:      {:.6f}".format(*hcv_score))

        print("test group:")
        hcv_score = homogeneity_completeness_v_measure(
            labels_true=reassign_noise(truth.particle_id, idx=(truth.particle_id == 0)),
            labels_pred=pred
        )
        print("homogeneity:    {:.6f}\n"
              "completeness:   {:.6f}\n"
              "v measure:      {:.6f}".format(*hcv_score))


def subroutine_2(hits, truth):
    print("=" * 120)
    aggregate_helix_1(hits.x, hits.y, hits.z, truth, merge_func=merge_two_2)


if __name__ == "__main__":
    print("start running script g1.py")
    s1 = Session(parent_dir="E:/TrackMLData/")
    for hits, truth in s1.remove_train_events(n=10, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        subroutine_2(hits, truth)
