
import pandas as pd
import numpy as np

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