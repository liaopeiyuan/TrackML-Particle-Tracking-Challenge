"""
g02.py
try cone scanning

Notes:
in clustering, cartesian coordinates (x/y) sometimes behave better than polar coordinate (angle/radius),
 because there's always a discontinuity between pi and -pi, 0 and 2pi, etc.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import scale
from sklearn.cluster import dbscan, DBSCAN

from geometric.display import plot_track_3d
from utils.session import Session
from geometric.utils import label_encode, reassign_noise, merge_discreet, merge_naive
from trackml.score import score_event


def subroutine_1(df):
    df = df.copy()
    # feature engineering
    df.loc[:, "rc"] = np.sqrt(df.x ** 2 + df.y ** 2)  # radius in cylindrical coordinate
    df.loc[:, "rs"] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)  # radius in spherical coordinate
    df.loc[:, "ac"] = np.arctan2(df.y, df.x)  # from -pi to pi
    df.loc[:, "az"] = np.arctan2(df.rc, df.z)  # from 0 to pi

    # df.loc[:, "u1"] = df.x / df.rs
    # df.loc[:, "u2"] = df.y / df.rs
    df.loc[:, "u3"] = df.z / df.rc

    pred = np.arange(df.shape[0])

    for az_center in np.arange(0, 181, 5):
        print("angle-z center: ", az_center)
        az_margin = 2
        az_center_rad, az_margin_rad = np.deg2rad(az_center), np.deg2rad(az_margin)

        lo, hi = az_center_rad - az_margin_rad, az_center_rad + az_margin_rad
        idx = (df.az >= lo) & (df.az < hi)

        def transform_dummy(sub_df):
            return scale(sub_df.loc[:, ["x", "y", "z"]])

        def transform_1(sub_df):
            # ret = scale(np.column_stack((sub_df.ac, sub_df.az, np.zeros(sub_df.shape[0]))))
            ret = np.column_stack([
                sub_df.x / sub_df.rc,
                sub_df.y / sub_df.rc,
                (sub_df.az - az_center_rad) / az_margin_rad
            ])
            # ret = scale(ret)
            return ret

        plot_track_3d(
            df.loc[idx, :],
            [transform_dummy, transform_1],
            [DBSCAN(eps=0.02, min_samples=1)], n_tracks=40)

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