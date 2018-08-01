"""
try to use machine learning algorithms to learn a best way to merge clusters
"""

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix
from itertools import combinations
from geometric.tools import reassign_noise

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


def get_pair_weight(weight):
    """
    the adjacency vector/matrix considers whether two points belong to the same particle, while this function computes
    how important that pair is by averaging the weights of the two individual points.
    """
    ret = weight + weight.reshape([-1, 1])
    ret = ret[np.tril_indices_from(ret, k=-1)] / 2  # flatten into vector
    return ret


def vector_to_symmetric_matrix(v):
    n = int((v.shape[0]*2) ** 0.5) + 1  # the shape of the symmetric matrix
    # this is the inverse formula from n*(n-1)//2
    ret = np.zeros([n, n])
    ret[np.tril_indices(n, -1)] = v
    ret += ret.T
    ret += np.identity(n)
    return ret


def prepare_bc_data(cluster_pred, particle_id=None, weight=None, binary_feature=False):
    """
    :param cluster_pred: (n_samples, n_steps) matrix
    :param particle_id: (n_samples,)
    :param weight: (n_samples,)
    :param binary_feature: use binary/bool as adjacency feature, rather than cluster size
    :return:
    prepare for binary classification
    """

    ret_w = None if weight is None else get_pair_weight(weight)
    ret_y = None if particle_id is None else get_flat_adjacency_vector(
        reassign_noise(particle_id, particle_id == 0)).astype(bool)
    # notice: noisy hits (particle_id == 0) will be reassigned to facilitate track size computation
    ret_x = csc_matrix((ret_y.shape[0], cluster_pred.shape[1]), dtype=(bool if binary_feature else float))
    for c in range(cluster_pred.shape[1]):
        ret_x[:, c] = get_flat_adjacency_vector(cluster_pred.iloc[:, c])
    # then, use a classifier such as logistic regression or lightgbm to fit (ret_x, ret_y, ret_w), ret_w is optional
    # even neural networks
    # tune hyperparameters
    return ret_x, ret_y, ret_w


def adjacency_pv_to_cluster_id(av, eps=0.5):
    """
    :param av: predicted adjacency probability vector from binary classifier
    :param eps: threshold to consider two points adjacent
    :return:
    """
    am = vector_to_symmetric_matrix(av > eps)  # predicted adjacency matrix
    n = am.shape[0]  # the number of samples
    # TODO: run DFS/BFS on am (adjacency matrix) with random initializations, essentially getting the clusters
    # TODO: let a(i, j) denote adjacency relationship provided by a binary classifier, I cannot guarantee that "a(i, j) and a(j, k) imply a(i, k)"
    # TODO: and that's why we need randomization: different random seeds may result in different clustering results

if __name__ == '__main__':
    mock_cluster_id = (np.random.rand(10) * 5).astype(int)
    # print(mock_cluster_id)
    # ret = get_flat_adjacency_vector(mock_cluster_id)
    # print(ret)
    r1 = get_pair_weight(np.array([1,2,3,4,5]))
    print(r1)
    r2 = vector_to_symmetric_matrix(r1)
    print(r2)


