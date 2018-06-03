"""
g04.py
cone slicing
by Tianyi Miao
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from trackml.score import score_event

from utils.session import Session
from geometric.g03 import RecursiveClusterer
from geometric.tools import merge_naive, reassign_noise, label_encode


def fast_score(df, pred):
    return score_event(
        truth=df,
        submission=pd.DataFrame({"hit_id": df.hit_id, "track_id": pred})
    )


def subroutine_show_psi(df):
    print("plot histogram for the distribution of psi")
    # from 0 to pi/2; closer to 0 -> closer to vertical track; closer to pi/2 -> closer to horizontal track
    df.loc[:, "psi"] = np.arctan2(np.sqrt(df.x ** 2 + df.y ** 2), np.abs(df.z))
    # all hits
    plt.hist(np.rad2deg(df.psi), bins=30, alpha=0.5, color='b', range=(0, 90.0))
    # non-noisy hits
    plt.hist(np.rad2deg(df.loc[df.particle_id != 0, "psi"]), bins=30, alpha=0.5, color="g", range=(0, 90.0))
    plt.xticks(np.linspace(0.0, 90.0, 10))
    plt.xlabel("angle r/z (degrees)")
    plt.title("Note: smaller angle corresponds to larger momentum in z-direction (tpz)")
    plt.show()


def subroutine_psi_slice(df, lo, hi):
    df.loc[:, "psi"] = np.arctan2(np.sqrt(df.x ** 2 + df.y ** 2), np.abs(df.z))
    idx = (df.psi >= np.deg2rad(lo)) & (df.psi < np.deg2rad(hi))
    best_cluster = label_encode(reassign_noise(df.particle_id, ~idx))
    best_score = fast_score(df, best_cluster)  # the best possible score achievable by the helix unrolling algorithm
    print("psi=[{}, {}), best possible score={:.6f}".format(lo, hi, best_score))
    h1 = RecursiveClusterer(
        p=2,
        dz0=-7e-4,
        stepdz=1e-5,
        eps0=0.0035,
        beta=0.5,
        max_step=140,
        feature_weight=np.array([1.0, 1.0, 0.75]),
        merge_func=lambda a, b: merge_naive(a, b, cutoff=20)
    )
    h1.fit_predict(df.loc[idx, :], score=True)


if __name__ == "__main__":
    print("start running script g04.py; cone slicing - exploration and running")
    np.random.seed()  # restart random number generator
    s1 = Session(parent_dir="E:/TrackMLData/")
    n_events = 20

    for hits, truth in s1.get_train_events(n=n_events, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        print("=" * 120)
        hits = hits.merge(truth, how="left", on="hit_id")
        subroutine_psi_slice(hits, 0, 10)


