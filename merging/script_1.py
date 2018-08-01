"""
test the clustering score from alex
"""

import numpy as np
import pandas as pd
from geometric.tools import merge_naive
from utils.session import Session

if __name__ == '__main__':
    np.random.seed(0)
    n_events = 20
    s1 = Session(parent_dir="E:/TrackMLData/")
    for x in s1.get_train_events(n=10, content=[s1.HITS, s1.TRUTH], randomness=True):
        print(x)
    print("bye")


