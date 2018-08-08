#get_ipython().run_line_magic('matplotlib', 'inline')
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


# In[2]:


def merge(cl1, cl2): # merge cluster 2 to cluster 1
    d = pd.DataFrame(data={'s1':cl1,'s2':cl2})
    d['N1'] = d.groupby('s1')['s1'].transform('count')
    d['N2'] = d.groupby('s2')['s2'].transform('count')
    maxs1 = d['s1'].max()+1
    cond = np.where((d['N2'].values>d['N1'].values) & (d['N2'].values<25)) # Locate the hit with the new cluster> old cluster
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


# In[4]:


if True: #tuning
    path_to_train = "/home/alexanderliao/data/Kaggle/competitions/trackml-particle-identification/train"
    event_prefix = "event000001555"
    hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))



# In[42]:


class Clusterer(object):
    def __init__(self):
        self.abc = []

    def initialize(self,dfhits):
        self.cluster = range(len(dfhits))

    def Clustering(self,
                   dfh,
                   coef,
                   epsilon,
                   min_samples=1,
                   n_loop=180,
                   stepii=0.000005,
                   dz=5.5,
                   step_dz=0.05,
                   verbose=True):
        merged_cluster = self.cluster
        mm = 1
        stepii = stepii
        count_ii = 0
        adaptive_eps_coefficient = 1
        dz=dz
        step_dz=step_dz



        #print(np.arange(0,step_dz,dz))
        for ii in np.arange(0, n_loop*stepii, stepii):
            for z0 in np.arange(0, dz, step_dz):
                count_ii += 1

                for jj in range(2):
                    mm = mm*(-1)
                    dfh=preprocess_hits(dfh, mm*z0)

                    eps_new = epsilon + count_ii*adaptive_eps_coefficient*10**(-5)

                    dfh['theta'] = dfh['phi'].values - np.nan_to_num(np.arccos(mm*ii*dfh['rt'].values))
                    dfh['sina1'] = np.sin(dfh['theta'].values)
                    dfh['cosa1'] = np.cos(dfh['theta'].values)

                    ss = StandardScaler()
                    dfs = ss.fit_transform(dfh[['sina1','cosa1','zdivrt','zdivr','xdivr','ydivr']].values)
                    #dfs = scale_ignore_nan(dfh[['sina1','cosa1','zdivrt','zdivr','xdivr','ydivr']])
                    dfs = np.multiply(dfs, coef)

                    new_cluster=DBSCAN(eps=eps_new,min_samples=min_samples,metric='euclidean',n_jobs=-1).fit(dfs).labels_
                    merged_cluster = merge(merged_cluster, new_cluster)


            if verbose == True:
                sub = create_one_event_submission(0, hits, merged_cluster)
                good_hits = extract_good_hits(truth, sub)
                score_1 = fast_score(good_hits)
                print('2r0_inverse:', ii*mm ,'. Score:', score_1)
                        #clear_output(wait=True)

            self.cluster = merged_cluster

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission

def preprocess_hits(h,dz):
    h['z'] =  h['z'].values + dz
    h['r'] = np.sqrt(h['x'].values**2+h['y'].values**2+h['z'].values**2)
    h['rt'] = np.sqrt(h['x'].values**2+h['y'].values**2)
    h['phi'] = np.arctan2(h['y'].values,h['x'].values)
    h['zdivrt'] = h['z'].values/h['rt'].values
    h['zdivr'] = h['z'].values/h['r'].values
    h['xdivr'] = h['x'].values / h['r'].values
    h['ydivr'] = h['y'].values / h['r'].values
    return h


# In[43]:


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


# In[44]:


# Clustering by varying
#model = Clusterer()
#model.initialize(hits)
#c = [1.5,1.5,0.73,0.17,0.027,0.027] 0.52
#c = [1.6,1.6,0.73,0.17,0.027,0.027] 0.529
#c = [1.45,1.45,0.73,0.17,0.027,0.027] 0.523

#[phi_coef,phi_coef,zdivrt_coef,zdivr_coef,xdivr_coef,ydivr_coef]
min_samples_in_cluster = 1
c = [1.6,1.6,0.73,0.17,0.027,0.027]
model = Clusterer()
model.initialize(hits)
hits_with_dz = preprocess_hits(hits, 0)
model.Clustering(hits_with_dz,coef=c,
                       epsilon=0.0048,
                       min_samples=1,
                       n_loop=180,
                       dz = 0.55,
                       step_dz = 0.05,
                       stepii=0.000005,
                       verbose=True)

submission = create_one_event_submission(0, hits, model.cluster)
print('\n')


# In[ ]:


# Clustering by varying
#model = Clusterer()
#model.initialize(hits)

# Preparing Submission
if False:
    for i in tqdm(range(62,125)):
        path_to_train = "/home/alexanderliao/data/Kaggle/competitions/trackml-particle-identification/test"
        event_prefix = "event"+str(i).zfill(9)
        hits = load_event_hits(os.path.join(path_to_train, event_prefix))
        c = [1.6,1.6,0.73,0.17,0.027,0.027] #[phi_coef,phi_coef,zdivrt_coef,zdivr_coef,xdivr_coef,ydivr_coef]
        min_samples_in_cluster = 1

        model = Clusterer()
        model.initialize(hits)
        hits_with_dz = preprocess_hits(hits, 0)
        model.Hough_clustering(hits_with_dz,coef=c,epsilon=0.0048,min_samples=min_samples_in_cluster,
                               n_loop=300,verbose=False)

        if i == 62:
            submission = create_one_event_submission(i, hits, model.cluster)
        else:
            submission = pd.concat([submission,create_one_event_submission(i, hits, model.cluster)])
        print(submission)
        if False: # O(n^2) if turned on
            submission.to_csv('submission.csv')
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
