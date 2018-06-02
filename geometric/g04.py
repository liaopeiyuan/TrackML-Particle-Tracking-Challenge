"""
g04.py
cone slicing
by Tianyi Miao
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.session import Session
from geometric.g03 import RecursiveClusterer
from geometric.tools import merge_naive


if __name__ == "__main__":
    print("start running script g04.py; cone slicing - exploration and running")
    np.random.seed()  # restart random number generator
    s1 = Session(parent_dir="E:/TrackMLData/")
    n_events = 20
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

    for hits, truth in s1.get_train_events(n=n_events, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        print("=" * 120)
        hits = hits.merge(truth, how="left", on="hit_id")

        hits.loc[:, "psi"] = np.arctan2(np.sqrt(hits.x ** 2 + hits.y ** 2), np.abs(hits.z))
        # from 0 to pi/2; closer to 0 -> closer to vertical track; closer to pi/2 -> closer to horizontal track
        plt.hist(np.rad2deg(hits.psi), bins=30)
        plt.show()

