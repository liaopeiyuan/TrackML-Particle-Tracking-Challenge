"""
Hough transform provided by Alex Liao on August 1st, 2018

"""

import tqdm
import pandas as pd
import numpy as np
from trackml.dataset import load_event_hits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import os


def merge(cl1, cl2):  # merge cluster 2 to cluster 1
    d = pd.DataFrame(data={'s1': cl1, 's2': cl2})
    d['N1'] = d.groupby('s1')['s1'].transform('count')
    d['N2'] = d.groupby('s2')['s2'].transform('count')
    maxs1 = d['s1'].max() + 1
    cond = np.where((d['N2'].values > d['N1'].values) & (d['N2'].values < 25))  # Locate the hit with the new cluster> old cluster
    s1 = d['s1'].values
    s1[cond] = d['s2'].values[cond] + maxs1  # Assign all hits that belong to the new track (+ maxs1 to increase the label for the track so it's different from the original).
    return s1


def extract_good_hits(truth, submission):
    tru = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    tru['count_both'] = tru.groupby(['track_id', 'particle_id']).hit_id.transform('count')
    tru['count_particle'] = tru.groupby(['particle_id']).hit_id.transform('count')
    tru['count_track'] = tru.groupby(['track_id']).hit_id.transform('count')
    return tru[(tru.count_both > 0.5 * tru.count_particle) & (tru.count_both > 0.5 * tru.count_track)]


def fast_score(good_hits_df):
    return good_hits_df.weight.sum()


def analyze_truth_perspective(truth, submission):
    tru = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    tru['count_both'] = tru.groupby(['track_id', 'particle_id']).hit_id.transform('count')
    tru['count_particle'] = tru.groupby(['particle_id']).hit_id.transform('count')
    tru['count_track'] = tru.groupby(['track_id']).hit_id.transform('count')
    good_hits = tru[(tru.count_both > 0.5 * tru.count_particle) & (tru.count_both > 0.5 * tru.count_track)]
    score = good_hits.weight.sum()
    anatru = tru.particle_id.value_counts().value_counts().sort_index().to_frame().rename(
        {'particle_id': 'true_particle_counts'}, axis=1)
    # anatru['true_particle_ratio'] = anatru['true_particle_counts'].values*100/np.sum(anatru['true_particle_counts'])
    anatru['good_tracks_counts'] = np.zeros(len(anatru)).astype(int)
    anatru['good_tracks_intersect_nhits_avg'] = np.zeros(len(anatru))
    anatru['best_detect_intersect_nhits_avg'] = np.zeros(len(anatru))
    for nhit in tqdm(range(4, 20)):
        particle_list = tru[(tru.count_particle == nhit)].particle_id.unique()
        if len(particle_list) == 0:
            continue
        intersect_count = 0
        good_tracks_count = 0
        good_tracks_intersect = 0
        for p in particle_list:
            nhit_intersect = tru[tru.particle_id == p].count_both.max()
            intersect_count += nhit_intersect
            corresponding_track = tru.loc[tru[tru.particle_id == p].count_both.idxmax()].track_id
            leng_corresponding_track = len(tru[tru.track_id == corresponding_track])
            if (nhit_intersect >= nhit / 2) and (nhit_intersect >= leng_corresponding_track / 2):
                good_tracks_count += 1
                good_tracks_intersect += nhit_intersect
        intersect_count = intersect_count / len(particle_list)
        anatru.at[nhit, 'best_detect_intersect_nhits_avg'] = intersect_count
        anatru.at[nhit, 'good_tracks_counts'] = good_tracks_count
        if good_tracks_count > 0:
            anatru.at[nhit, 'good_tracks_intersect_nhits_avg'] = good_tracks_intersect / good_tracks_count
    return score, anatru, good_hits


def precision(truth, submission, min_hits):
    tru = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    tru['count_both'] = tru.groupby(['track_id', 'particle_id']).hit_id.transform('count')
    tru['count_particle'] = tru.groupby(['particle_id']).hit_id.transform('count')
    tru['count_track'] = tru.groupby(['track_id']).hit_id.transform('count')
    # print('Analyzing predictions...')
    predicted_list = tru[(tru.count_track >= min_hits)].track_id.unique()
    good_tracks_count = 0
    ghost_tracks_count = 0
    fp_weights = 0
    tp_weights = 0
    for t in predicted_list:
        nhit_track = tru[tru.track_id == t].count_track.iloc[0]
        nhit_intersect = tru[tru.track_id == t].count_both.max()
        corresponding_particle = tru.loc[tru[tru.track_id == t].count_both.idxmax()].particle_id
        leng_corresponding_particle = len(tru[tru.particle_id == corresponding_particle])
        if (nhit_intersect >= nhit_track / 2) and (
            nhit_intersect >= leng_corresponding_particle / 2):  # if the predicted track is good
            good_tracks_count += 1
            tp_weights += tru[(tru.track_id == t) & (tru.particle_id == corresponding_particle)].weight.sum()
            fp_weights += tru[(tru.track_id == t) & (tru.particle_id != corresponding_particle)].weight.sum()
        else:  # if the predicted track is bad
            ghost_tracks_count += 1
            fp_weights += tru[(tru.track_id == t)].weight.sum()
    all_weights = tru[(tru.count_track >= min_hits)].weight.sum()
    precision = tp_weights / all_weights * 100
    print('Precision: ', precision, ', good tracks:', good_tracks_count, ', total tracks:', len(predicted_list),
          ', loss:', fp_weights, ', reco:', tp_weights, 'reco/loss', tp_weights / fp_weights)
    return precision


class Clusterer(object):
    def __init__(self):
        self.abc = []
        self.cluster_pred = []

    def Hough_clustering(self, dfh, coef, epsilon, min_samples=1, n_loop=225, verbose=True):
        # merged_cluster = self.cluster
        mm = 1
        stepii = 0.000004
        count_ii = 0
        adaptive_eps_coefficient = 1
        for ii in np.arange(0, n_loop * stepii, stepii):
            count_ii += 1
            for jj in range(2):
                mm = mm * (-1)
                eps_new = epsilon + count_ii * adaptive_eps_coefficient * 10 ** (-5)
                dfh['a1'] = dfh['a0'].values - np.nan_to_num(np.arccos(mm * ii * dfh['rt'].values))
                dfh['sina1'] = np.sin(dfh['a1'].values)
                dfh['cosa1'] = np.cos(dfh['a1'].values)
                ss = StandardScaler()
                dfs = ss.fit_transform(dfh[['sina1', 'cosa1', 'zdivrt', 'zdivr', 'xdivr', 'ydivr']].values)
                # dfs = scale_ignore_nan(dfh[['sina1','cosa1','zdivrt','zdivr','xdivr','ydivr']])
                dfs = np.multiply(dfs, coef)
                new_cluster = DBSCAN(eps=eps_new, min_samples=min_samples, metric='euclidean', n_jobs=-1).fit_predict(dfs)
                self.cluster_pred.append(new_cluster)
                # merged_cluster = merge(merged_cluster, new_cluster)
        # self.cluster = merged_cluster


def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id] * len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission


def preprocess_hits(h, dz):
    h['z'] = h['z'].values + dz
    h['r'] = np.sqrt(h['x'].values ** 2 + h['y'].values ** 2 + h['z'].values ** 2)
    h['rt'] = np.sqrt(h['x'].values ** 2 + h['y'].values ** 2)
    h['a0'] = np.arctan2(h['y'].values, h['x'].values)
    h['zdivrt'] = h['z'].values / h['rt'].values
    h['zdivr'] = h['z'].values / h['r'].values
    h['xdivr'] = h['x'].values / h['r'].values
    h['ydivr'] = h['y'].values / h['r'].values
    return h


# In[4]:


# Clustering by varying
# model = Clusterer()
# model.initialize(hits)
"""
if __name__ == '__main__':
    from utils.session import Session
    s1 = Session()
    temp_data = []
    for hits, truth in s1.get_train_events(n=3, content=[s1.HITS, s1.TRUTH], randomness=False)[1]: temp_data.append([hits[["x", "y", "z"]], truth["particle_id"], truth["weight"]])
    
    for i in tqdm(range(62, 125)):
        path_to_train = "/home/alexanderliao/data/Kaggle/competitions/trackml-particle-identification/test"
        event_prefix = "event" + str(i).zfill(9)
        hits = load_event_hits(os.path.join(path_to_train, event_prefix))
        c = [1.6, 1.6, 0.73, 0.17, 0.027, 0.027]  # [phi_coef,phi_coef,zdivrt_coef,zdivr_coef,xdivr_coef,ydivr_coef]
        min_samples_in_cluster = 1

        model = Clusterer()
        model.initialize(hits)
        hits_with_dz = preprocess_hits(hits, 0)
        model.Hough_clustering(hits_with_dz, coef=c, epsilon=0.0048, min_samples=min_samples_in_cluster, n_loop=300, verbose=False)

        if i == 62:
            submission = create_one_event_submission(i, hits, model.cluster)
        else:
            submission = pd.concat([submission, create_one_event_submission(i, hits, model.cluster)])
        print(submission)
        # submission.to_csv('submission.csv')
    print('\n')
    submission.to_csv('submission.csv')
"""


def dfh_gen_1(df, coef, n_steps=225, mm=1, stepii=4e-6):
    df = df.copy()
    # df['z'] = df['z'] + dz # TODO: the r later may be different
    df['r'] = np.sqrt(df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2)
    df['rt'] = np.sqrt(df['x'] ** 2 + df['y'] ** 2)
    df['a0'] = np.arctan2(df['y'], df['x'])
    df['zdivrt'] = df['z'] / df['rt']
    df['zdivr'] = df['z'] / df['r']
    df['xdivr'] = df['x'] / df['r']
    df['ydivr'] = df['y'] / df['r']
    for ii in np.arange(0, n_steps * stepii, stepii):
        for jj in range(2):
            mm = mm * (-1)
            df['a1'] = df['a0'].values - np.nan_to_num(np.arccos(mm * ii * df['rt'].values))
            df['sina1'] = np.sin(df['a1'])
            df['cosa1'] = np.cos(df['a1'])
            ss = StandardScaler()
            dfs = ss.fit_transform(df[['sina1', 'cosa1', 'zdivrt', 'zdivr', 'xdivr', 'ydivr']].values)
            # dfs = scale_ignore_nan(dfh[['sina1','cosa1','zdivrt','zdivr','xdivr','ydivr']])
            dfs = np.multiply(dfs, coef)
            yield dfs

    
def clusterer_gen_1(n_steps=225, adaptive_eps_coef=1, eps=0.05, min_samples=1, metric="euclidean", p=2, n_jobs=1):
    for ii in range(1, n_steps + 1):
        for jj in range(2):
            eps_new = eps + ii * adaptive_eps_coef * 1e-5
            yield DBSCAN(eps=eps_new, min_samples=min_samples, n_jobs=n_jobs, metric=metric, metric_params=None, p=p)

c = [1.6, 1.6, 0.73, 0.17, 0.027, 0.027]
import multiprocessing as mp
p1 = mp.Pool()
hits = pd.DataFrame()
temp_data = [["hits", "particle_id", "weight"], ["hits", "particle_id", "weight"], ["hits", "particle_id", "weight"]]
def pred_wrapper(arg):
    return arg[1].fit_predict(arg[0])
p1 = mp.Pool(processes=12)
cluster_pred_0 = list(p1.map(pred_wrapper, zip(dfh_gen_1(temp_data[0][0], coef=c, n_steps=225, mm=1, stepii=4e-6), clusterer_gen_1(225, adaptive_eps_coef=1, eps=0.0048, min_samples=1, metric="euclidean", p=2, n_jobs=1))))
cluster_pred_1 = list(p1.map(pred_wrapper, zip(dfh_gen_1(temp_data[1][0], coef=c, n_steps=225, mm=1, stepii=4e-6), clusterer_gen_1(225, adaptive_eps_coef=1, eps=0.0048, min_samples=1, metric="euclidean", p=2, n_jobs=1)), chunksize=15))
cluster_pred_2 = list(p1.map(pred_wrapper, zip(dfh_gen_1(temp_data[2][0], coef=c, n_steps=225, mm=1, stepii=4e-6), clusterer_gen_1(225, adaptive_eps_coef=1, eps=0.0048, min_samples=1, metric="euclidean", p=2, n_jobs=1)), chunksize=15))
# cluster_pred_1 = list(map(lambda arg: arg[1].fit_predict(arg[0]), zip(dfh_gen_1(hits, coef=c, n_steps=225, mm=1, stepii=4e-6), clusterer_gen_1(225, adaptive_eps_coef=1, eps=0.0048, min_samples=1, metric="euclidean", p=2, n_jobs=-1))))
# should give a list of cluster ids, each a n_samples length array
