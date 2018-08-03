```import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats
from tqdm import tqdm
from trackml.dataset import load_event, load_dataset
from trackml.score import score_event
import os

def analyze_truth_perspective(truth, submission):
    tru = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    tru['count_both'] = tru.groupby(['track_id', 'particle_id']).hit_id.transform('count')    
    tru['count_particle'] = tru.groupby(['particle_id']).hit_id.transform('count')
    tru['count_track'] = tru.groupby(['track_id']).hit_id.transform('count')
    good_hits = tru[(tru.count_both > 0.5*tru.count_particle) & (tru.count_both > 0.5*tru.count_track)]
    score = good_hits.weight.sum()
    
    anatru = tru.particle_id.value_counts().value_counts().sort_index().to_frame().rename({'particle_id':'true_particle_counts'}, axis=1)
    #anatru['true_particle_ratio'] = anatru['true_particle_counts'].values*100/np.sum(anatru['true_particle_counts'])

    anatru['good_tracks_counts'] = np.zeros(len(anatru)).astype(int)
    anatru['good_tracks_intersect_nhits_avg'] = np.zeros(len(anatru))
    anatru['best_detect_intersect_nhits_avg'] = np.zeros(len(anatru))
    for nhit in tqdm(range(4,20)):
        particle_list  = tru[(tru.count_particle==nhit)].particle_id.unique()
        intersect_count = 0
        good_tracks_count = 0
        good_tracks_intersect = 0
        for p in particle_list:
            nhit_intersect = tru[tru.particle_id==p].count_both.max()
            intersect_count += nhit_intersect
            corresponding_track = tru.loc[tru[tru.particle_id==p].count_both.idxmax()].track_id
            leng_corresponding_track = len(tru[tru.track_id == corresponding_track])
            
            if (nhit_intersect >= nhit/2) and (nhit_intersect >= leng_corresponding_track/2):
                good_tracks_count += 1
                good_tracks_intersect += nhit_intersect
        intersect_count = intersect_count/len(particle_list)
        anatru.at[nhit,'best_detect_intersect_nhits_avg'] = intersect_count
        anatru.at[nhit,'good_tracks_counts'] = good_tracks_count
        if good_tracks_count > 0:
            anatru.at[nhit,'good_tracks_intersect_nhits_avg'] = good_tracks_intersect/good_tracks_count
    
    return score, anatru, good_hits

def precision(truth, submission,min_hits):
    tru = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    tru['count_both'] = tru.groupby(['track_id', 'particle_id']).hit_id.transform('count')    
    tru['count_particle'] = tru.groupby(['particle_id']).hit_id.transform('count')
    tru['count_track'] = tru.groupby(['track_id']).hit_id.transform('count')
    #print('Analyzing predictions...')
    predicted_list  = tru[(tru.count_track>=min_hits)].track_id.unique()
    good_tracks_count = 0
    ghost_tracks_count = 0
    fp_weights = 0
    tp_weights = 0
    for t in predicted_list:
        nhit_track = tru[tru.track_id==t].count_track.iloc[0]
        nhit_intersect = tru[tru.track_id==t].count_both.max()
        corresponding_particle = tru.loc[tru[tru.track_id==t].count_both.idxmax()].particle_id
        leng_corresponding_particle = len(tru[tru.particle_id == corresponding_particle])
        if (nhit_intersect >= nhit_track/2) and (nhit_intersect >= leng_corresponding_particle/2): #if the predicted track is good
            good_tracks_count += 1
            tp_weights += tru[(tru.track_id==t)&(tru.particle_id==corresponding_particle)].weight.sum()
            fp_weights += tru[(tru.track_id==t)&(tru.particle_id!=corresponding_particle)].weight.sum()
        else: # if the predicted track is bad
                ghost_tracks_count += 1
                fp_weights += tru[(tru.track_id==t)].weight.sum()
    all_weights = tru[(tru.count_track>=min_hits)].weight.sum()
    precision = tp_weights/all_weights*100
    print('Precision: ',precision,', good tracks:', good_tracks_count,', total tracks:',len(predicted_list),
           ', loss:', fp_weights, ', reco:', tp_weights, 'reco/loss', tp_weights/fp_weights)
    return precision


def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission  

def merge(cl1, cl2): # merge cluster 2 to cluster 1
    d = pd.DataFrame(data={'s1':cl1,'s2':cl2})
    d['N1'] = d.groupby('s1')['s1'].transform('count')
    d['N2'] = d.groupby('s2')['s2'].transform('count')
    maxs1 = d['s1'].max()+1
    cond = np.where((d['N2'].values>d['N1'].values) & (d['N2'].values<23)) # Locate the hit with the new cluster> old cluster
    s1 = d['s1'].values 
    s1[cond] = d['s2'].values[cond]+maxs1 # Assign all hits that belong to the new track (+ maxs1 to increase the label for the track so it's different from the original).
    return s1

def extract_good_hits(truth, submission):
    tru = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    tru['count_both'] = tru.groupby(['track_id', 'particle_id']).hit_id.transform('count')    
    tru['count_particle'] = tru.groupby(['particle_id']).hit_id.transform('count')
    tru['count_track'] = tru.groupby(['track_id']).hit_id.transform('count')
    return tru[(tru.count_both > 0.5*tru.count_particle) & (tru.count_both > 0.5*tru.count_track)]

def fast_score(good_hits_df):
    return good_hits_df.weight.sum()


def df_gen_1(df, coef, n_steps=225, mm=1, stepii=4e-6,z0=0):
    """
    default code provided by Alex on Slack, August 2nd
    """
    df['z'] = df['z'] + z0  # TODO: the r later may be different
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
            df['a1'] = df['a0'].values - np.arccos(mm*ii*df['rt'].values)
            cond=np.where(np.isfinite(df['a1'].values))
            df['sina1'] = np.zeros(len(df))
            df['cosa1'] = np.zeros(len(df))
            df['sina1'].values[cond] = np.sin(df['a1'].values[cond])
            df['cosa1'].values[cond] = np.cos(df['a1'].values[cond])
            ss = StandardScaler()
            dfs = ss.fit_transform(df[['sina1', 'cosa1', 'zdivrt', 'zdivr', 'xdivr', 'ydivr']].values)
            # dfs = scale_ignore_nan(df[['sina1','cosa1','zdivrt','zdivr','xdivr','ydivr']])
            dfs = np.multiply(dfs, coef)
            yield dfs


def clusterer_gen_1(hits, truth, n_steps=300, adaptive_eps_coef=1, eps=0.0048, min_samples=1, metric="euclidean", p=2, n_jobs=1, verbose=False):
    """
    default code provided by Alex on Slack, August 2nd
    """
    for ii in range(1, n_steps + 1):
        for jj in range(2):
            eps_new = eps + ii * adaptive_eps_coef * 1e-5
            cluster = DBSCAN(eps=eps_new, min_samples=min_samples, n_jobs=n_jobs, metric=metric, metric_params=None, p=p)
            if verbose == True:
                sub = create_one_event_submission(0, hits, cluster)
                good_hits = extract_good_hits(truth, sub)
                score_1 = fast_score(good_hits)
                print('2r0_inverse:', ii ,'. Score:', score_1)
            yield cluster


def run_helix_cluster(df_gen, clusterer_gen, parallel=False):
    def pred_wrapper(arg):
        return arg[1].fit_predict(arg[0])
        
    if parallel:
        pool_1 = mp.Pool()
        return list(pool_1.map(pred_wrapper, zip(df_gen, clusterer_gen)))
    else:
        return list(map(pred_wrapper, zip(df_gen, clusterer_gen)))


if __name__ == '__main__':
    path_to_train = "/home/alexanderliao/data/Kaggle/competitions/trackml-particle-identification/train"
    event_prefix = "event000001000"
    hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))
    c = [1.5,1.5,0.73,0.17,0.027,0.027] #[phi_coef,phi_coef,zdivrt_coef,zdivr_coef,xdivr_coef,ydivr_coef]
    min_samples_in_cluster = 1

    predictions = run_helix_cluster(df_gen_1(hits, coef=c, n_steps=225, mm=1, stepii=4e-6,z0=0),clusterer_gen_1(hits, truth, n_steps=300, adaptive_eps_coef=1, eps=0.0048, min_samples=1, metric="euclidean", p=2, n_jobs=6,verbose=True),parallel=True)

    submission = create_one_event_submission(0, hits, predictions)
    pr = precision(truth,submission,min_hits=10)
    pr = precision(truth,submission,min_hits=7)
    pr = precision(truth,submission,min_hits=4)
    pr = precision(truth,submission,min_hits=1)```