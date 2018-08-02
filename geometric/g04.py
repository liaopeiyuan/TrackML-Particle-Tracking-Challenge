"""
g04.py
cone slicing
by Tianyi Miao
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.preprocessing import scale
from sklearn.cluster import dbscan

from trackml.score import score_event

from utils.session import Session
from geometric.helix import HelixUnroll
from geometric.tools import merge_naive, reassign_noise, label_encode, hit_completeness


class HelixUnrollWithRadius(HelixUnroll):
    def fit_predict(self, df, score_func=None, verbose=False):
        print("method override: helix unroll with radius")
        df = df.copy()
        df.loc[:, "r3"] = self.r3_func(df.x, df.y, df.z)

        # df.loc[:, "rs"] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)  # radius in spherical coordinate system
        df.loc[:, "rc"] = np.sqrt(df.x ** 2 + df.y ** 2)  # radius in cylindrical coordinate system
        df.loc[:, "a0"] = np.arctan2(df.y, df.x)  # angle in cylindrical coordinate system
        df.loc[:, "z1"] = df.z / df.rc  # monotonic with cot(psi)

        pred = None
        score_list = []

        for i in range(self.n_steps):
            dz = self.dz_func(i)
            df.loc[:, "a1"] = df.a0 + dz * df.r3  # rotation, points with higher r3 are rotated to a larger degree
            # convert angle to sin/cos -> more intuitive in Euclidean distance
            # e.g. 2pi and 0 should be very close
            df.loc[:, "sina1"] = np.sin(df.a1)
            df.loc[:, "cosa1"] = np.cos(df.a1)
            df.loc[:, "z_new"] = df.z1  # old HelixUnroll method
            # df.loc[:, "z_new"] = df.z / (df.rc + dz * df.r3 * 500)

            # scale the space
            dfs = scale(df.loc[:, ["sina1", "cosa1", "z_new"]])
            # use hidden transformation methods to re-weight the features. Consider nonlinear transformations later.
            dfs = self.hidden_transform(dfs)
            res = \
            dbscan(X=dfs, eps=self.eps_func(i), min_samples=1, metric="minkowski", p=self.p, n_jobs=self.dbscan_n_jobs)[1]
            pred = self.merge_func(pred, res)

            if score_func is not None:
                # use a callback to customize scoring
                step_score = score_func(pred)
                score_list.append(step_score)
                if verbose:
                    print(str(i).rjust(3) + ": {:.6f}".format(step_score))
        return pred, np.array(score_list)


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

    def hidden_transform_temp(x):
        x[:, 0] *= 1.0
        x[:, 1] *= 1.0
        x[:, 2] *= 0.4
        return x

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

    hu_20_50 = HelixUnroll(
        r3_func=lambda x, y, z: np.sqrt(x ** 2 + y ** 2 + z ** 2),
        dz_func=lambda i: (-1) ** (i + 1) * (-7e-4 + i * 1e-5),
        n_steps=400,  # TODO
        hidden_transform=lambda x: x * np.array([1.1, 1.1, 0.3]),
        merge_func=merge_naive,
        eps_func=lambda i: 3.5e-3 + 5e-6 * i,
        p=2,
        dbscan_n_jobs=-1
    )

    hu_70_90 = HelixUnroll(
        r3_func=lambda x, y, z: np.sqrt(x ** 2 + y ** 2 + z ** 2),
        dz_func=lambda i: (-1) ** (i + 1) * (-7e-4 + 1e-5 * i),
        n_steps=1000,
        hidden_transform=lambda x: x * np.array([1.2, 1.2, 0.25]),
        merge_func=merge_naive,
        eps_func=lambda i: 3.5e-3 + 5e-6 * i,
        p=2,
        dbscan_n_jobs=-1
    )

    hu_now = HelixUnrollWithRadius(
        r3_func=lambda x, y, z: np.sqrt(x ** 2 + y ** 2 + z ** 2),
        dz_func=lambda i: (-1) ** (i + 1) * (-7e-4 + i * 1e-5),
        n_steps=160,
        hidden_transform=lambda x: x * np.array([1.2, 1.2, 0.35]),
        merge_func=merge_naive,
        eps_func=lambda i: 3.5e-3 + 5e-6 * i,
        p=2,
        dbscan_n_jobs=-1
    )

    def temp_score_func(pred):
        full_pred = best_cluster.copy()
        full_pred[idx] = pred
        return fast_score(df, full_pred)  # / best_score

    hu_now.fit_predict(df.loc[idx, :], score_func=temp_score_func, verbose=True)


if __name__ == "__main__":
    print("start running script g04.py; cone slicing - exploration and running")
    np.random.seed()  # restart random number generator
    s1 = Session(parent_dir="E:/TrackMLData/")
    n_events = 20

    for hits, truth in s1.get_train_events(n=n_events, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        print("=" * 120)
        hits = hits.merge(truth, how="left", on="hit_id")
        subroutine_psi_slice(hits, 60, 90)

