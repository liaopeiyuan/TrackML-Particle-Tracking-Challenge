"""
utils.py

useful tools for clustering
"""
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


def label_encode(y):
    return LabelEncoder().fit_transform(y)


def reassign_noise(labels: np.ndarray, idx):
    """
    assign noisy points (labeled with key_value such as -1 or 0) to their own clusters of size 1
    """
    ret = labels.copy()
    ret[idx] = np.arange(np.sum(idx)) + np.max(ret) + 1
    return ret


def merge_naive(pred_1, pred_2, cutoff=20):
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


def merge_discreet(pred_1, pred_2, cutoff=21):
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


def hit_completeness(df, idx, track_size):
    """
    (the number of non-noisy hits in the idx) / (the total number of hits from all particles
    that have at least 1 hit in the idx)
    """
    num = (df.loc[idx, "particle_id"] != 0).sum()
    all_particles = df.loc[idx, "particle_id"].unique().tolist()
    if 0 in all_particles:
        all_particles.remove(0)
    denom = track_size[all_particles].sum()
    return num / denom


def track_completeness(df, idx):
    """
    (number of tracks with all hits in the region) / (number of tracks that have at least 1 hit in the region)
    idx is a boolean mask over the region
    """
    all_particles = df.loc[idx, "particle_id"].unique().tolist()
    if 0 in all_particles:
        all_particles.remove(0)

    agg_1 = df.loc[idx, :].groupby("particle_id", sort=True)["x"].agg("count")
    if 0 in agg_1:
        agg_1.drop(0, inplace=True)
    agg_2 = df.loc[df.particle_id.isin(all_particles), :].groupby("particle_id", sort=True)["x"].agg("count")
    return np.mean(agg_1 == agg_2)
