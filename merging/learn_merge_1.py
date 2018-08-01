"""
try to use machine learning algorithms to learn a best way to merge clusters
"""

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from itertools import combinations


def get_flat_adjacency_vector(cluster_id):
    """
    For n samples, an adjacency matrix is a symmetric matrix with shape (n_samples, n_samples) indicating whether two
    points are adjacent to each other. In the context of TrackML, two hits are adjacent iff they are caused by the same
    particle.

    Because an adjacency matrix is symmetric, and its diagonal is always 1 (a hit is adjacent to itself), we take only
    its lower triangle:
    (0, 0), (0, 1), (0, 2), (0, 3)              (1, 0)
    (1, 0), (1, 1), (1, 2), (1, 3)      ==>     (2, 0), (2, 1)
    (2, 0), (2, 1), (2, 2), (2, 3)              (3, 0), (3, 1), (3, 2)
    (3, 0), (3, 1), (3, 2), (3, 3)
    and flatten it to an adjacency vector av:
    (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)
    Note: (i, j) indicates the adjacency between the ith and the jth sample; A[i, j] == A[j, i]; A[i, i] == 1;
    A[i, j] == 1 or A[i, j] == 0

    Therefore, the length of the adjacency vector will be n*(n-1)//2
    To access A[i, j], assume WLOG that i > j, A[i, j] = i*(i-1)//2+j

    Because of the small track sizes, most elements of av is 0. Because av is a column vector (and will later be stacked
    horizontally into a design matrix), I will use a sparse csc matrix to store it. It is more memory-efficient than
    a dense boolean array, let alone that the array may not be boolean.

    :param cluster_id: 1d pd.Series or np.array with shape (n_samples,)
    """
    n = cluster_id.shape[0]
    ret = np.zeros(n*(n-1)//2)  # column vector as dense array
    pred = pd.DataFrame({"cluster_id": cluster_id})
    pred = pred.join(pred["cluster_id"].value_counts().rename("cluster_size"), on="cluster_id")  # get cluster size
    pred["sample_index"] = pred.index
    pred = pred.loc[pred["cluster_size"] > 1, :]  # eliminate singletons to save groupby time

    def subroutine(sub_df):
        cluster_size = sub_df["cluster_size"].iloc[0]  # cluster size is the same for all points in a cluster
        for j, i in combinations(sub_df["sample_index"], r=2):  # j < i is guaranteed
            ret[i*(i-1)//2+j] = cluster_size
            # if adjacent, return positive cluster size, which provides more information than binary indicator
            # could be useful for machine learning algorithms
    pred.groupby("cluster_id").agg(subroutine)  # use agg instead of apply to avoid running the first group twice
    return ret


def adjacency_vector_to_adjacency_matrix(av):
    pass


if __name__ == '__main__':
    mock_cluster_id = (np.random.rand(10) * 5).astype(int)
    print(mock_cluster_id)
    ret = get_flat_adjacency_vector(mock_cluster_id)
    print(ret)