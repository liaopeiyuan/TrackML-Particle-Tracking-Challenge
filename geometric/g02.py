"""
g02.py
try cone scanning
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import scale
from sklearn.cluster import dbscan, DBSCAN

from eda_old.display import plot_track_3d
from geometric.session import Session
from geometric.utils import label_encode, reassign_noise, merge_discreet, merge_naive
from trackml.score import score_event


def transform_1(df):
    return df[["ac", "az", "rc"]]


def subroutine_1(df):
    # feature engineering
    df.loc[:, "rc"] = np.sqrt(df.x ** 2 + df.y ** 2)  # radius in cylindrical coordinate
    df.loc[:, "rs"] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)  # radius in spherical coordinate
    df.loc[:, "ac"] = np.arctan2(df.y, df.x)  # from -pi to pi
    df.loc[:, "az"] = np.arctan2(df.rc, df.z)  # from 0 to pi

    df.loc[:, "v1"] = df.x / df.rs
    df.loc[:, "v2"] = df.y / df.rs
    df.loc[:, "v3"] = df.z / df.rc

    pred = np.arange(df.shape[0])

    for az_center in np.arange(0, 181, 5):
        print("angle-z center: ", az_center)
        az_margin = 2
        lo, hi = np.deg2rad(az_center - az_margin), np.deg2rad(az_center + az_margin)
        idx = (df.az >= lo) & (df.az < hi)

        plot_track_3d(df.loc[idx, :], [transform_1], [DBSCAN(eps=0.008, min_samples=1)])

        if False:
            res = -np.ones(df.shape[0])
            res[idx] = dbscan(X=scale(df.loc[idx, ["v1", "v2", "v3"]].values), eps=0.0075, min_samples=1, n_jobs=-1)[1]
            res = reassign_noise(res, res == -1)
            pred = merge_discreet(pred, res, cutoff=20)

            official_score = score_event(
                truth=df,
                submission=pd.DataFrame({"hit_id": df.hit_id, "track_id": pred})
            )
            print("official score: {:.6f}".format(official_score))


if __name__ == "__main__":
    print("start running script g02.py")
    s1 = Session(parent_dir="E:/TrackMLData/")
    for hits, truth in s1.remove_train_events(n=10, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        print("=" * 120)
        hits = hits.merge(truth, how="left", on="hit_id")
        subroutine_1(hits)