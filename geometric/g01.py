"""
g01.py

first geometric script

by Tianyi Miao

some notes:
official_score = score_event(truth=truth, submission=pd.DataFrame({"hit_id": truth.hit_id, "track_id": pred}))
hcv_score = homogeneity_completeness_v_measure(labels_true=reassign_noise(truth.particle_id), labels_pred=pred)

"""

from collections import Counter

import numpy as np
import pandas as pd

from sklearn.preprocessing import scale, LabelEncoder
from sklearn.cluster import dbscan

from geometric.session import Session

from trackml.score import score_event
from sklearn.metrics import homogeneity_completeness_v_measure


def label_encode(y):
    return LabelEncoder().fit_transform(y)


def helix_1(x, y, z):
    """
    helix unrolling that works since the beginning
    """
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    return x/d, y/d, z/r


def reassign_noise(labels: np.ndarray, key_value=0):
    """
    assign noisy points (labeled with key_value such as -1 or 0) to their own clusters of size 1
    """
    ret = labels.copy()
    idx = (ret == key_value)
    ret[idx] = np.array(list(range(int(np.sum(idx))))) + np.max(ret) + 1
    return ret


def merge2_func1(pred_1, pred_2):
    c1, c2 = Counter(pred_1), Counter(pred_2)  # track id -> track size
    n1, n2 = np.vectorize(c1.__getitem__)(pred_1), np.vectorize(c2.__getitem__)(pred_2)  # hit id -> track size
    pred = pred_1.copy()
    pred[(n2 > n1)] = max(pred_1) + 1 + pred_2[(n2 > n1)]
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
            labels_true=reassign_noise(truth.particle_id, key_value=0),
            labels_pred=pred
        )
        print("homogeneity:    {:.6f}\n"
              "completeness:   {:.6f}\n"
              "v measure:      {:.6f}".format(*hcv_score))


def subroutine_2(hits, truth):
    pass



if __name__ == "__main__":
    print("start running script g1.py")
    s1 = Session(parent_dir="E:/TrackMLData/")
    for hits, truth in s1.remove_train_events(n=10, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        subroutine_2(hits, truth)