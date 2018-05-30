"""
g02.py
try cone scanning
"""

from geometric.session import Session
from geometric.utils import label_encode, reassign_noise, merge_discreet, merge_naive


def subroutine_1(df):
    pass

if __name__ == "__main__":
    print("start running script g1.py")
    s1 = Session(parent_dir="E:/TrackMLData/")
    for hits, truth in s1.remove_train_events(n=10, content=[s1.HITS, s1.TRUTH], randomness=True)[1]:
        print("=" * 120)
        hits = hits.merge(truth, how="left", on="hit_id")
        subroutine_1(hits)