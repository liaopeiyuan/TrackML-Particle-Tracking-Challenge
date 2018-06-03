"""
helix.py

algorithms related to helix unrolling

by Tianyi Miao
"""

import numpy as np
import pandas as pd

from sklearn.cluster import dbscan
from sklearn.preprocessing import scale

from geometric.tools import merge_naive


class HelixUnroll(object):
    def __init__(
            self,
            # helix-unrolling parameters
            r3_func=lambda x, y, z: np.sqrt(x**2 + y**2 + z**2),
            dz_func=lambda i: (-1)**(i+1) * (-7e-4 + i * 1e-5),
            n_steps=150,
            hidden_transform=lambda x: x * np.array([1.0, 1.0, 0.75]),
            # cluster aggregation parameters
            merge_func=merge_naive,
            # dbscan parameters
            eps_func=lambda i: 3.5e-3 + 5e-6 * i,
            p=2,
            dbscan_n_jobs=-1
            ):
        self.r3_func = r3_func
        self.dz_func = dz_func
        self.n_steps = n_steps
        self.hidden_transform = hidden_transform  # transform the hidden space after scaling, before dbscan

        self.merge_func = merge_func

        self.eps_func = eps_func
        self.p = p
        self.dbscan_n_jobs = dbscan_n_jobs

    def fit_predict(self, df, score_func=None, verbose=False):
        df = df.copy()
        df.loc[:, "r3"] = self.r3_func(df.x, df.y, df.z)

        # df.loc[:, "rs"] = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2)  # radius in spherical coordinate system
        df.loc[:, "rt"] = np.sqrt(df.x ** 2 + df.y ** 2)  # radius in cylindrical coordinate system
        df.loc[:, "a0"] = np.arctan2(df.y, df.x)  # angle in cylindrical coordinate system
        df.loc[:, "z1"] = df.z / df.rt  # monotonic with cot(psi)
        # df.loc[:, "z2"] = df.z / df.rs TODO: 4 feature maybe? [1, 1, 0.4, 0.4]

        pred = None
        score_list = []

        for i in range(self.n_steps):
            dz = self.dz_func(i)
            df.loc[:, "a1"] = df.a0 + dz * df.r3  # rotation, points with higher r3 are rotated to a larger degree
            # convert angle to sin/cos -> more intuitive in Euclidean distance
            # e.g. 2pi and 0 should be very close
            df.loc[:, "sina1"] = np.sin(df.a1)
            df.loc[:, "cosa1"] = np.cos(df.a1)
            # scale the space
            dfs = scale(df.loc[:, ["sina1", "cosa1", "z1"]])
            # use hidden transformation methods to re-weight the features. Consider nonlinear transformations later.
            dfs = self.hidden_transform(dfs)
            res = dbscan(X=dfs, eps=self.eps_func(i), min_samples=1, metric="minkowski", p=self.p, n_jobs=self.dbscan_n_jobs)[1]
            pred = self.merge_func(pred, res)

            if score_func is not None:
                # use a callback to customize scoring
                step_score = score_func(pred)
                score_list.append(step_score)
                if verbose:
                    print(str(i).rjust(3) + ": {:.6f}".format(step_score))
        return pred, np.array(score_list)


if __name__ == "__main__":
    h2 = HelixUnroll(
        r3_func=lambda x, y, z: np.sqrt(x ** 2 + y ** 2 + z ** 2),
        dz_func=lambda i: (-1) ** (i + 1) * (-7e-4 + i * 1e-5),
        n_steps=120,
        hidden_transform=lambda x: x * np.array([1.0, 1.0, 0.75]),
        merge_func=merge_naive,
        eps_func=lambda i: 3.5e-3 + 5e-6 * i,
        p=2,
        dbscan_n_jobs=-1
    )
    # this benchmark version of HelixUnroll can give a score up to 0.468
    # I save the hyperparameters here. Do not change.

