
# coding: utf-8

# In this notebook, I would like to use the Hough transform to cluster hits. This notebook is therefore, get some materials from the past published kernels,
# 
# In the previous notebook, we see a function relating phi and r like (where $r = \sqrt{x^2 + y^2}$, $\phi = arctan2(y/x)$):
# $$ \phi_{new} = \phi + i(ar + br^2),$$
# where $i$ is increased incrementally from 0 (straight tracks) to some number (curve tracks).
# 
# 
# However, the above equation is not exact to relate those two features. Instead, one might want to use the Hough transform:
# $$  \frac{r}{2r_0} =  \cos(\phi - \theta) $$
# 
# In the above equation, $\phi$ and $r$ are the original $\phi$ and $r$ of each hit, while $r_0$ and $\theta$ are the $r$ and $\phi$ of a specific point in the XY plane, that is the origin of a circle in XY plane. That circle passes through the inspected hit. 
# 
# Then, our clustering problem can be stated this way:
# - For each $\frac{1}{2r_0}$, starting from 0 (corresponding to straight tracks), to an appropriate stopping point, we calculate $\theta = \phi - \arccos(\frac{r}{2r_0})$
# - Group all hits with the near $\theta$ and some other features to a detected track by DBSCAN. Since $\theta$ can take very large or small values, using $\sin(\theta)$ and $\cos(\theta)$ is better.
# 

# In[1]:


import multiprocessing as mp

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
#import hdbscan
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import DBSCAN

from IPython.display import clear_output

from trackml_helper import *
from analysis import *
import random


# In[2]:


def merge(cl1, cl2): # merge cluster 2 to cluster 1
    d = pd.DataFrame(data={'s1':cl1,'s2':cl2})
    d['N1'] = d.groupby('s1')['s1'].transform('count')
    d['N2'] = d.groupby('s2')['s2'].transform('count')
    maxs1 = d['s1'].max()
    cond = np.where((d['N2'].values>d['N1'].values) & (d['N2'].values<19)) # Locate the hit with the new cluster> old cluster
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


# In[3]:


if False: #tuning
    path_to_train = "/home/alexanderliao/data/Kaggle/competitions/trackml-particle-identification/train"
    event_prefix = "event000001000"
    hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))
    


# In[4]:




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


# In[5]:


def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission  

def preprocess_hits(h,dz):
    h['z'] =  h['z'].values + dz
    h['r'] = np.sqrt(h['x'].values**2+h['y'].values**2+h['z'].values**2)
    h['rt'] = np.sqrt(h['x'].values**2+h['y'].values**2)
    h['a0'] = np.arctan2(h['y'].values,h['x'].values)
    h['zdivrt'] = h['z'].values/h['rt'].values
    h['zdivr'] = h['z'].values/h['r'].values
    h['xdivr'] = h['x'].values / h['r'].values
    h['ydivr'] = h['y'].values / h['r'].values
    return h


# In[6]:


def smart_arccos(x):
    max_mask = x > 1
    min_mask = x < -1
    ret = np.arccos(x, where=~(max_mask|min_mask))
    ret[max_mask] = 0.0
    ret[min_mask] = np.pi
    return np.pi

class Clusterer(object):
    def __init__(self):                        
        self.abc = []
          
    def initialize(self,dfhits):
        self.cluster = range(len(dfhits))
        
    def Hough_clustering(self,dfh,coef,epsilon,min_samples=1,n_loop=180,verbose=True): # [phi_coef,phi_coef,zdivrt_coef,zdivr_coef,xdivr_coef,ydivr_coef]
        merged_cluster = self.cluster
        mm = 1
        stepii = 0.000005
        count_ii = 0
        adaptive_eps_coefficient = 1
        #z = np.arange(-5.5,5.5,0.01)
        #random.choice(z)
        for ii in np.arange(0, n_loop*stepii, stepii):
            count_ii += 1
            for jj in range(2):
                mm = mm*(-1)
                eps_new = epsilon + count_ii*adaptive_eps_coefficient*10**(-5)
                #eps_new = 0.0035
                
                """
                dfh['a1'] = dfh['a0'].values - smart_arccos(mm*ii*dfh['rt'].values)
                dfh['sina1']= np.sin(dfh['a1'].values)
                dfh['cosa1']= np.cos(dfh['a1'].values)
                
                """
                dfh['a1'] = dfh['a0'].values - np.arccos(mm*ii*dfh['rt'].values)
                cond=np.where(np.isfinite(dfh['a1'].values))
                dfh['sina1'] = np.random.rand(len(dfh))
                dfh['cosa1'] = np.random.rand(len(dfh))
                dfh['sina1'].values[cond] = np.sin(dfh['a1'].values[cond])
                dfh['cosa1'].values[cond] = np.cos(dfh['a1'].values[cond])
                
                
                ss = StandardScaler()
                dfs = ss.fit_transform(dfh[['sina1','cosa1','zdivrt','zdivr','xdivr','ydivr']].values) 
                #dfs = scale_ignore_nan(dfh[['sina1','cosa1','zdivrt','zdivr','xdivr','ydivr']])
                dfs = np.multiply(dfs, coef)
                new_cluster=DBSCAN(eps=eps_new,min_samples=min_samples,metric='euclidean',n_jobs=-1).fit(dfs).labels_
                
                cond=np.where(np.bincount(new_cluster)>45)
                new_cluster[cond] = np.random.randint(low=max(new_cluster),size=len(cond))
                
                merged_cluster = merge(merged_cluster, new_cluster)
                
                if verbose == True:
                    sub = create_one_event_submission(0, hits, merged_cluster)
                    good_hits = extract_good_hits(truth, sub)
                    score_1 = fast_score(good_hits)
                    print('2r0_inverse:', ii*mm ,'. Score:', score_1)
                    #clear_output(wait=True)
        self.cluster = merged_cluster                 


# In[7]:


# Clustering by varying 
#model = Clusterer()
#model.initialize(hits) 
if False:
    c = [1.5,1.5,0.73,0.17,0.027,0.027] #[phi_coef,phi_coef,zdivrt_coef,zdivr_coef,xdivr_coef,ydivr_coef]
    min_samples_in_cluster = 1

    model = Clusterer()
    model.initialize(hits) 
    hits_with_dz = preprocess_hits(hits, 0)
    model.Hough_clustering(hits_with_dz,coef=c,epsilon=0.0048,min_samples=min_samples_in_cluster,
                           n_loop=300,verbose=True)

    submission = create_one_event_submission(0, hits, model.cluster)
    print('\n')


# In[8]:


pr = precision(truth,submission,min_hits=10)


# In[9]:


pr = precision(truth,submission,min_hits=7)


# In[10]:


pr = precision(truth,submission,min_hits=4)


# In[11]:


pr = precision(truth,submission,min_hits=1)


# In[12]:


if False: #benchmark
    c = [1.6,1.6,0.73,0.17,0.027,0.027]
    model = Clusterer()
    model.initialize(hits) 
    hits_with_dz = preprocess_hits(hits, 0)
    model.Clustering(hits,coef=c,
                           epsilon=0.0048,
                           min_samples=1,
                           n_loop=180,
                           stepii=0.000005,
                           verbose=True)


# In[ ]:


# Clustering by varying 
#model = Clusterer()
#model.initialize(hits) 

# Preparing Submission
if True:
    for i in tqdm(range(125)):
        path_to_train = "/home/alexanderliao/data/Kaggle/competitions/trackml-particle-identification/test"
        event_prefix = "event"+str(i).zfill(9)
        hits = load_event_hits(os.path.join(path_to_train, event_prefix))
        c=[1.5,1.5,0.73,0.17,0.027,0.027] #[phi_coef,phi_coef,zdivrt_coef,zdivr_coef,xdivr_coef,ydivr_coef]
        min_samples_in_cluster = 1

        model = Clusterer()
        model.initialize(hits) 
        hits_with_dz = preprocess_hits(hits, 0)
        model.Hough_clustering(hits_with_dz,coef=c,epsilon=0.0048,min_samples=min_samples_in_cluster,
                               n_loop=300,verbose=False)

        if i == 0:
            submission = create_one_event_submission(i, hits, model.cluster)
        else:
            submission = pd.concat([submission,create_one_event_submission(i, hits, model.cluster)])
        print(submission)
        if i%15==0: # O(n^2) if turned on
            submission.to_csv('submission-temp.csv')
    print('\n') 
    submission.to_csv('submission.csv')


# In[ ]:


submission
df = submission.track_id.unique()


# In[ ]:


6967634/len(df)


# In[ ]:


for i in range(63,124):
    print(i)


# Now, let us see some analysis on the clustering result:

# In[ ]:


pr = precision(truth,submission,min_hits=10)


# In[ ]:


pr = precision(truth,submission,min_hits=7)


# In[ ]:


pr = precision(truth,submission,min_hits=4)


# In[ ]:


pr = precision(truth,submission,min_hits=1)


# As one can see, long tracks have high precision, low lost weights. On the other hand, there are too many ghost short tracks. Then, we can use multi-stage clustering, using min_hits in DBSCAN for each stage (i.e., cluster long tracks first, then cluster short tracks with different parameters without touching the long tracks...).

# Some other notes:
# 
# + Use too many loops can decrease the performance, as one can see from the log result above.
# 
# + No z-shifting is performed  (dz = 0), although the function preprocess already offer it. Some may want to use z-shifting right away just by change dz from 0 to any number between [-5.5, 5.5]
# 
# + Features are not optimized. Honestly, I am also stuck at searching for good features (and good weights). It would be very nice if someone secretly tell me those magic features :-).
# 
# + When $r/(2r_0) > 1$ or $< 1$, arccos is undefined, hence a warning appears (if running on local notebook). The problem, more importantly, is not about the warning. It is a technical issue: all hits with $r/(2r_0) > 1$ or $< 1$ MUST BE EXCLUDED from DBSCAN, because there will be NO track with that parameter pass through those hits. This can be done by some indexing techniques that I do not provide here. (DBSCAN uses a raw matrix to cluster, then we must be careful when exclude hits from the original full hit dataframe).
# 
# KV.
