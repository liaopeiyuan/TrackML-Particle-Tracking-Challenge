"""
d01.py

explore geometric properties of particle tracks

by Tianyi Miao

Notes:

"""

from geometric.session import Session

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def subroutine_1(df):
    """
    explore track size
    """
    print("="*120)
    track_size = df.groupby("particle_id")["x"].agg("count")
    track_size.drop(0, axis=0, inplace=True)
    plt.hist(track_size, alpha=0.3, color="g", range=(0.5, 21.5), bins=21)
    plt.xticks(range(0, 21))


def subroutine_2(df, n=20):
    """
    plot 3d tracks
    """
    track_size = df.groupby("particle_id")["x"].agg("count")
    track_size.drop(0, axis=0, inplace=True)
    track_size[track_size > 3].index


if __name__ == "__main__":
    print("start running script d01.py")

    s1 = Session(parent_dir="E:/TrackMLData/")
    for hits, truth in s1.remove_train_events(n=15, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        hits = hits.merge(truth, how="left", on="hit_id")
        subroutine_1(hits[["x", "y", "z", "particle_id", "weight"]])
    plt.show()