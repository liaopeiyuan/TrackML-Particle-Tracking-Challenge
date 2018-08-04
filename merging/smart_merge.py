"""
try to use machine learning algorithms to learn a best way to merge clusters
"""
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, dok_matrix, hstack as sparse_hstack, csr_matrix
import networkx as nx
from itertools import combinations
from geometric.tools import reassign_noise

'''
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
    raise DeprecationWarning("this version is deprecated for memory issues")
    n = cluster_id.shape[0]
    ret = np.zeros(n * (n - 1) // 2)  # column vector as dense array
    pred = pd.DataFrame({"cluster_id": cluster_id})
    pred = pred.join(pred["cluster_id"].value_counts().rename("cluster_size"), on="cluster_id")  # get cluster size
    pred["sample_index"] = pred.index
    pred = pred.loc[pred["cluster_size"] > 1, :]  # eliminate singletons to save groupby time
    
    def subroutine(sub_df):
        cluster_size = sub_df["cluster_size"].iloc[0]  # cluster size is the same for all points in a cluster
        for j, i in combinations(sub_df["sample_index"], r=2):  # j < i is guaranteed
            ret[i * (i - 1) // 2 + j] = cluster_size
            # if adjacent, return positive cluster size, which provides more information than binary indicator
            # could be useful for machine learning algorithms
            
    pred.groupby("cluster_id").agg(subroutine)  # use agg instead of apply to avoid running the first group twice
    return ret
'''

def get_flat_adjacency_vector(cluster_id):
    n = cluster_id.shape[0]
    ret = dok_matrix((n*(n-1)//2, 1), dtype=np.uint8)
    pred = pd.DataFrame({"cluster_id": cluster_id})
    pred = pred.join(pred["cluster_id"].value_counts().rename("cluster_size"), on="cluster_id")  # get cluster size
    pred["sample_index"] = pred.index
    pred = pred.loc[pred["cluster_size"] > 1, :]  # eliminate singletons to save groupby time
    def subroutine(sub_df):
        cluster_size = sub_df["cluster_size"].iloc[0]  # cluster size is the same for all points in a cluster
        for j, i in combinations(sub_df["sample_index"], r=2):  # j < i is guaranteed
            ret[i * (i - 1) // 2 + j, 0] = cluster_size
            # if adjacent, return positive cluster size, which provides more information than binary indicator
            # could be useful for machine learning algorithms
    pred.groupby("cluster_id").agg(subroutine)  # use agg instead of apply to avoid running the first group twice
    return ret
    
'''
def get_pair_weight(weight):
    """
    the adjacency vector/matrix considers whether two points belong to the same particle, while this function computes
    how important that pair is by averaging the weights of the two individual points.
    """
    weight = np.array(weight)
    ret = weight + weight.reshape([-1, 1])
    ret = ret[np.tril_indices_from(ret, k=-1)]  # flatten into vector
    ret = ret / np.sum(ret)
    return ret
'''


def get_pair_weight(idx, weight):
    # idx is a 2d array: n_samples * 2
    weight = np.array(weight)
    ret_w = weight[idx[:, 0]] + weight[idx[:, 1]]
    ret_w = ret_w * (len(ret_w) / np.sum(ret_w))
    return ret_w


"""
def vector_to_symmetric_matrix(v):
    raise DeprecationWarning("This function 'vector_to_symmetric_matrix' is deprecated except for testing purposes, "
                             "please refer to adjacency_pv_to_cluster_id instead.")
    n = int((v.shape[0] * 2) ** 0.5) + 1  # the shape of the symmetric matrix
    # this is the inverse formula from n*(n-1)//2
    ret = np.zeros([n, n])
    ret[np.tril_indices(n, -1)] = v
    ret += ret.T
    ret += np.identity(n)
    return ret

"""


def get_bc_data(cluster_pred, particle_id=None, weight=None, binary_feature=False, parallel=True):
    """
    :param cluster_pred: list (len = n_steps) of cluster id prediction arrays (length n_samples)
    :param particle_id: (n_samples,)
    :param weight: (n_samples,)
    :param binary_feature: use binary/bool as adjacency feature, rather than cluster size
    :return:
    prepare for binary classification
    """
    n = len(cluster_pred[0])
    print("Preparing ret_x")
    ret_x = sparse_hstack(blocks=list(
        mp.Pool().map(get_flat_adjacency_vector, cluster_pred) if parallel else
        map(get_flat_adjacency_vector, cluster_pred)
    ),
        format="csr", dtype=(bool if binary_feature else np.uint8))
    print("Preparing mask")
    # if True in mask, the two points are assigned to the same cluster in at least one step
    mask = ret_x.indptr[1:] != ret_x.indptr[:-1]  # if True in mask, will be kept in memory
    ret_x = ret_x[mask, :]
    print("Preparing idx")
    ret_idx = np.array(np.tril_indices(n, -1)).T[mask]
    print("Preparing ret_w")
    ret_w = None if weight is None else get_pair_weight(idx=ret_idx, weight=weight)
    print("Preparing ret_y")
    ret_y = None if particle_id is None else get_flat_adjacency_vector(reassign_noise(particle_id, particle_id == 0)).astype(bool)[mask].toarray().ravel()
    # notice: noisy hits (particle_id == 0) will be reassigned to facilitate track size computation
    # then, use a classifier such as lr/lgb/nn to fit (ret_x, ret_y, ret_w), ret_w is optional
    print("Done")
    return ret_idx, ret_x.astype(np.float32), ret_y, ret_w


def adjacency_pv_to_cluster_id(n, idx, apv, eps=0.5):
    """
    :param apv: predicted adjacency probability vector from binary classifier (only the ones in mask)
    :param eps: threshold to consider two points adjacent
    :param mask: boolean mask with shape n*(n-1)//2
    :return:
    """
    # n = int((apv.shape[0] * 2) ** 0.5) + 1  # the shape of the symmetric matrix
    # this is the inverse formula from n*(n-1)//2
    apv = np.array(apv).ravel()
    g1 = nx.from_edgelist(idx[apv > eps].tolist())
    c_id = 1
    ret = np.zeros(n)
    for component in nx.connected_components(g1):
        ret[list(component)] = c_id
        c_id += 1
    return ret


def predict_bc(cluster_pred, predict_func, eps=0.5, binary_feature=False, parallel=True):
    # predict_func must return a probability vector
    idx, x, _, _ = get_bc_data(cluster_pred, binary_feature=binary_feature, parallel=parallel)
    ret = adjacency_pv_to_cluster_id(cluster_pred.shape[0], idx, predict_func(x), eps)
    return ret
    

def temp_main(binary_classifier, train_data, val_data, score_func):
    for cluster_pred, particle_id, weight in train_data:  # train_data is a list of 3-tuples
        idx, x, y, w = get_bc_data(cluster_pred, particle_id, weight, binary_feature=False)
        binary_classifier.fit(x, y, w)  # it's also possible to see validation score here if you use lightgbm
    for cluster_pred, particle_id, weight in val_data:  # train_data is a list of 3-tuples
        idx, x, y, w = get_bc_data(cluster_pred, binary_feature=False)
        apv = binary_classifier.predict(x)
        final_pred = adjacency_pv_to_cluster_id(apv, eps=0.5)
        print(score_func(final_pred, particle_id, weight))


if __name__ == '__main__':
    print("hello, world")
    # mock_cluster_id = (np.random.rand(10) * 5).astype(int)
    # print(mock_cluster_id)
    # ret = get_flat_adjacency_vector(mock_cluster_id)
    # print(ret)
    # r1 = get_pair_weight(np.array([1, 2, 3, 4, 5]))
    # r2 = get_pair_weight_fast([1, 2, 3, 4, 5])
    # print(r1)
    # print(r2)
    # r2 = vector_to_symmetric_matrix(r1)
    # print(r2)
