"""
test the clustering score from alex
"""

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from geometric.tools import merge_naive
from utils.session import Session


class HelixUnroll3(object):
    """
    (Tianyi) this is my 3rd version of HelixUnroll, which seeks to emulate that by Alex Liao
    """



if __name__ == '__main__':
    np.random.seed(0)
    n_events = 20
    s1 = Session(parent_dir="E:/TrackMLData/")
    for x in s1.get_train_events(n=10, content=[s1.HITS, s1.TRUTH], randomness=True):
        print(x)
    print("bye")


