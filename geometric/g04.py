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
from geometric.helix import HelixUnroll
from geometric.tools import merge_naive, reassign_noise, label_encode, hit_completeness


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

    hu_0_10 = HelixUnroll(
        r3_func=lambda x, y, z: np.sqrt(x ** 2 + y ** 2 + z ** 2),
        dz_func=lambda i: (-1) ** (i + 1) * (-7e-4 + i * 1e-5),
        n_steps=120,
        hidden_transform=lambda x: x * np.array([1.1, 1.1, 0.6]),
        merge_func=merge_naive,
        eps_func=lambda i: 3.5e-3 + 5e-6 * i,
        p=2,
        dbscan_n_jobs=-1
    )

    hu_50_70 = HelixUnroll(
        r3_func=lambda x, y, z: np.sqrt(x ** 2 + y ** 2 + z ** 2),
        dz_func=lambda i: (-1) ** (i + 1) * (-7e-4 + i * 1e-5),
        n_steps=120,
        hidden_transform=lambda x: x * np.array([1.1, 1.1, 0.6]),
        merge_func=merge_naive,
        eps_func=lambda i: 3.5e-3 + 5e-6 * i,
        p=2,
        dbscan_n_jobs=-1
    )

    hu_70_90 = HelixUnroll(
        r3_func=lambda x, y, z: np.sqrt(x ** 2 + y ** 2 + z ** 2),
        dz_func=lambda i: (-1) ** (i + 1) * (-7e-4 + 1e-5 * i),
        n_steps=1000,
        hidden_transform=lambda x: x * np.array([1.2, 1.2, 0.1]),
        merge_func=merge_naive,
        eps_func=lambda i: 3.5e-3 + 5e-6 * i,
        p=2,
        dbscan_n_jobs=-1
    )

    def temp_score_func(pred):
        full_pred = best_cluster.copy()
        full_pred[idx] = pred
        return fast_score(df, full_pred)  # / best_score

    hu_50_70.fit_predict(df.loc[idx, :], score_func=temp_score_func, verbose=True)


if __name__ == "__main__":
    print("start running script g04.py; cone slicing - exploration and running")
    np.random.seed()  # restart random number generator
    s1 = Session(parent_dir="/home/alexanderliao/data/GitHub/Kaggle-TrackML/portable-dataset/")
    n_events = 20

    for hits, truth in s1.get_train_events(n=n_events, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        print("=" * 120)
        hits = hits.merge(truth, how="left", on="hit_id")
        subroutine_show_psi(hits)
        subroutine_psi_slice(hits, 50, 90)




