import numpy as np
import pandas as pd
from tqdm import trange
import multiprocessing as mp

from functools import reduce
from itertools import product

from utils.session import Session
from finalAtom.utils.clusterer import Clusterer
from trackml.score import score_event

s1 = Session("../data/")

r1_cluster_pred = []
r1_truth = []
for hits, truth in s1.get_train_events(n=1)[1]:
    res = Clusterer(None, None, None).predict_merge(hits)
    r1_cluster_pred.append([x[0] for x in res])
    r1_truth.append(truth)
    
    
def pairwise_merge(pred_1, pred_2, cutoff=20):
    if pred_1 is None:
        return pred_2
    d = pd.DataFrame(data={'s1': pred_1, 's2': pred_2})
    d['N1'] = d.groupby('s1')['s1'].transform('count')
    d['N2'] = d.groupby('s2')['s2'].transform('count')
    max_s1 = d['s1'].max() + 1
    cond = np.where((d['N2'].values > d['N1'].values) & (d['N2'].values < cutoff))
    s1 = d['s1'].values
    s1[cond] = d['s2'].values[cond] + max_s1
    return s1


def full_merge(cluster_pred, ordering=None):
    return reduce(pairwise_merge, cluster_pred if ordering is None else (cluster_pred[i] for i in ordering))

    
def easy_score(truth, pred):
    return score_event(
        truth=truth,
        submission=pd.DataFrame({"hit_id": truth.hit_id, "track_id": pred})
    )


def temp_wrapper_3(idx):
    return easy_score(r1_truth[0], full_merge(r1_cluster_pred[0], idx))


def temp_wrapper_3_1(idx):
    return easy_score(r1_truth[1], full_merge(r1_cluster_pred[1], idx))


r2_idx = []
r2_score = []

with trange(3000) as t:
    for i in t:
        max_score = max(r2_score) if r2_score else -1.0
        t.set_postfix(max_score=f"{max_score:.6}")
        with mp.Pool() as p1:
            idx_list = [np.random.permutation(range(len(r1_cluster_pred[0]))) for j in range(20)]
            score_list = list(p1.map(temp_wrapper_3, idx_list))
            r2_idx.extend(idx_list)
            r2_score.extend(score_list)
        cond = np.argsort(r2_score)[-200:][::-1]  # sort in descending scores
        r2_score = [r2_score[j] for j in cond]
        r2_idx = [r2_idx[j] for j in cond]
        

for d in (400, 500):
    with trange(20) as t:
        idx = r2_idx[0]
        for i in t:  # iterate over current best permutations
                t.set_postfix(max_score=f"{max(r2_score):.5}")
                idx_list = []
                for j in range(25):  # permutations of current mask
                    new_idx = idx.copy()
                    perm_mask = np.random.choice(range(len(idx)), size=len(idx) // d, replace=False)
                    new_idx[perm_mask] = new_idx[perm_mask][np.random.permutation(range(len(idx) // d))]
                    idx_list.append(new_idx)
                with mp.Pool() as p1:
                    score_list = list(p1.map(temp_wrapper_3, idx_list))
                    r2_idx.extend(idx_list)
                    r2_score.extend(score_list)
                cond = np.argsort(r2_score)[-200:][::-1]  # sort in descending scores
                r2_score = [r2_score[i] for i in cond]
                r2_idx = [r2_idx[i] for i in cond]

'''
def distribution_summary(cluster_id, k, le=True, average_cluster=True):
    """
    calculate a summary statistic for the value counts of a cluster_id
    :param k: a threshold value
    :param le: if True, return the average for the tracks with size less than or equal to k;
        otherwise, return the average for the tracks with size greater than or equal to k
    :param average_cluster: if True, average the counts weighted by the number of clusters (equivalent to count);
        otherwise, average the counts weighted by the number of hits
    """
    vc = pd.value_counts(cluster_id, sort=False, normalize=False)
    return np.average(((vc <= k) if le else (vc >= k)), weights=(None if average_cluster else vc.index))


def temp_wrapper_4(args):
    k, le, average_cluster, ascending = args
    cluster_lek_mean = list(map(lambda x: distribution_summary(x, k=k, le=le, average_cluster=average_cluster), r1_cluster_pred[0]))
    idx = np.argsort(cluster_lek_mean).tolist()
    if not ascending:
        idx = idx[::-1]
    return temp_wrapper_3(idx)


r3_params = list(product(range(1, 100), [False, True], [False, True], [False, True]))
r3_score = []
parallel_size = 25
with trange(7, len(r3_params) // parallel_size + 1) as t:
    for i in t:
        max_score = max(r3_score) if r3_score else -1.0
        t.set_postfix(max_score=f"{max_score:.6}")
        with mp.Pool() as p1:
            r3_score.extend(p1.map(temp_wrapper_4, r3_params[i*parallel_size:(i+1)*parallel_size]))

            
            
    
'''

# benchmark score for another event
temp_wrapper_3_1(None)
temp_wrapper_3_1(r2_idx[0])
temp_wrapper_3_1(r2_idx[-1])


for i in range(1):
    easy_score(r1_truth[i], full_merge(r1_cluster_pred[i]))

