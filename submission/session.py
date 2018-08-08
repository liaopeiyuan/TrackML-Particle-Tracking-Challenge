"""
session.py
by Tianyi Miao abnd Alexander Liao

- class
Session

- main

"""

import os

import numpy as np
import pandas as pd

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

import matplotlib.pyplot as plt
import os
import gc
from sklearn.preprocessing import StandardScaler
import hdbscan
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.cluster.dbscan_ import dbscan

from sklearn.neighbors import KDTree
import timeit
import multiprocessing
from multiprocessing import Pool


class Session(object):
    """
    A highly integrated framework for efficient data loading, prediction submission, etc. in TrackML Challenge
    (improved version of the official TrackML package)

    Precondition: the parent directory must be organized as follows:
    - train (directory)
        - event000001000-cells.csv
        ...
    - test (directory)
        - event000000001-cells.csv
        ...
    - detectors.csv
    - sample_submission.csv
    """
    # important constants to avoid spelling errors
    HITS = "hits"
    CELLS = "cells"
    PARTICLES = "particles"
    TRUTH = "truth"
    
    def __init__(self, parent_dir="./", train_dir="train/", test_dir="test/", detectors_dir="detectors.csv",
                 sample_submission_dir="sample_submission.csv",rz_scales=[0.65, 0.965, 1.528]):
        """
        default input:
        Session("./", "train/", "test/", "detectors.csv", "sample_submission.csv")
        Session(parent_dir="./", train_dir="train/", test_dir="test/", detectors_dir="detectors.csv", sample_submission_dir="sample_submission.csv")
        """
        self._parent_dir = parent_dir
        self._train_dir = train_dir
        self._test_dir = test_dir
        self._detectors_dir = detectors_dir
        self._sample_submission_dir = sample_submission_dir
        self.rz_scales = rz_scales
        
        if not os.path.isdir(self._parent_dir):
            raise ValueError("The input parent directory {} is invalid.".format(self._parent_dir))
        
        # there are 8850 events in the training dataset; some ids from 1000 to 9999 are skipped
        if os.path.isdir(self._parent_dir + self._train_dir):
            self._train_event_id_list = sorted(
                set(int(x[x.index("0"):x.index("-")]) for x in os.listdir(self._parent_dir + self._train_dir)))
        else:
            self._train_dir = None
            self._train_event_id_list = []
        
        if os.path.isdir(self._parent_dir + self._test_dir):
            self._test_event_id_list = sorted(
                set(int(x[x.index("0"):x.index("-")]) for x in os.listdir(self._parent_dir + self._test_dir)))
        else:
            self._test_dir = None
            self._test_event_id_list = []
        
        if not os.path.exists(self._parent_dir + self._detectors_dir):
            self._detectors_dir = None
        
        if not os.path.exists(self._parent_dir + self._sample_submission_dir):
            self._sample_submission_dir = None
    
    @staticmethod
    def get_event_name(event_id):
        return "event" + str(event_id).zfill(9)
    
    def get_train_events(self, n=10, content=(HITS, TRUTH), randomness=True):
        n = min(n, len(self._train_event_id_list))
        if randomness:
            event_ids = np.random.choice(self._train_event_id_list, size=n, replace=False).tolist()
        else:
            event_ids = self._train_event_id_list[:n]
            self._train_event_id_list = self._train_event_id_list[n:] + self._train_event_id_list[:n]
        
        event_names = [Session.get_event_name(event_id) for event_id in event_ids]
        return event_names, \
               (load_event(self._parent_dir + self._train_dir + event_name, content) for event_name in event_names)
    
    def remove_train_events(self, n=10, content=(HITS, TRUTH), randomness=True):
        """
        get n events from self._train_event_id_list:
        if random, get n random events; otherwise, get the first n events
        :return:
         1. ids: event ids
         2. an iterator that loads a tuple of hits/cells/particles/truth files
        remove these train events from the current id list
        """
        n = min(n, len(self._train_event_id_list))
        if randomness:
            event_ids = np.random.choice(self._train_event_id_list, size=n, replace=False).tolist()
            for event_id in event_ids:
                self._train_event_id_list.remove(event_id)
        else:
            event_ids, self._train_event_id_list = self._train_event_id_list[:n], self._train_event_id_list[n:]
        
        event_names = [Session.get_event_name(event_id) for event_id in event_ids]
        return event_names, \
               (load_event(self._parent_dir + self._train_dir + event_name, content) for event_name in event_names)
    
    def get_test_event(self, n=3, content=(HITS, TRUTH), randomness=True):
        n = min(n, len(self._test_event_id_list))
        if randomness:
            event_ids = np.random.choice(self._test_event_id_list, size=n, replace=False).tolist()
        else:
            event_ids, = self._test_event_id_list[:n]
            self._test_event_id_list = self._test_event_id_list[n:] + self._test_event_id_list[:n]
        
        event_names = [Session.get_event_name(event_id) for event_id in event_ids]
        return event_names, \
               (load_event(self._parent_dir + self._test_dir + event_name, content) for event_name in event_names)
    
    def remove_test_events(self, n=10, content=(HITS, CELLS), randomness=False):
        n = min(n, len(self._test_event_id_list))
        if randomness:
            event_ids = np.random.choice(self._test_event_id_list, size=n, replace=False).tolist()
            for event_id in event_ids:
                self._test_event_id_list.remove(event_id)
        else:
            event_ids, self._test_event_id_list = self._test_event_id_list[:n], self._test_event_id_list[n:]
        event_names = [Session.get_event_name(event_id) for event_id in event_ids]
        return event_names, \
               (load_event(self._parent_dir + self._test_dir + event_name, content) for event_name in event_names)
    
    def _extend(self,submission,hits,limit=0.04, num_neighbours=18):
        df = submission.merge(hits,  on=['hit_id'], how='left')
        df = df.assign(d = np.sqrt( df.x**2 + df.y**2 + df.z**2 ))
        df = df.assign(r = np.sqrt( df.x**2 + df.y**2))
        df = df.assign(arctan2 = np.arctan2(df.z, df.r))

        #for angle in range( -90,90,1):
        #ytt
        for angle in np.arange( -90,90,0.5):

            print ('\r %f'%angle, end='',flush=True)
            #df1 = df.loc[(df.arctan2>(angle-0.5)/180*np.pi) & (df.arctan2<(angle+0.5)/180*np.pi)] #bad
            df1 = df.loc[(df.arctan2>(angle-1.5)/180*np.pi) & (df.arctan2<(angle+1.5)/180*np.pi)]

            min_num_neighbours = len(df1)
            if min_num_neighbours<3: continue

            hit_ids = df1.hit_id.values
            x,y,z = df1[['x', 'y', 'z']].values.T
            r  = (x**2 + y**2)**0.5
            r  = r/1000
            a  = np.arctan2(y,x)
            c = np.cos(a)
            s = np.sin(a)
            #tree = KDTree(np.column_stack([a,r]), metric='euclidean')
            tree = KDTree(np.column_stack([c, s, r]), metric='euclidean')


            track_ids = list(df1.track_id.unique())
            num_track_ids = len(track_ids)
            #min_length=3
            #ytt
            min_length=3

            for i in range(num_track_ids):
                p = track_ids[i]
                if p==0: continue

                idx = np.where(df1.track_id==p)[0]
                if len(idx)<min_length: continue

                if angle>0:
                    idx = idx[np.argsort( z[idx])]
                else:
                    idx = idx[np.argsort(-z[idx])]


                ## start and end points  ##
                idx0,idx1 = idx[0],idx[-1]
                a0 = a[idx0]
                a1 = a[idx1]
                r0 = r[idx0]
                r1 = r[idx1]
                c0 = c[idx0]
                c1 = c[idx1]
                s0 = s[idx0]
                s1 = s[idx1]

                da0 = a[idx[1]] - a[idx[0]]  #direction
                dr0 = r[idx[1]] - r[idx[0]]
                direction0 = np.arctan2(dr0,da0)

                da1 = a[idx[-1]] - a[idx[-2]]
                dr1 = r[idx[-1]] - r[idx[-2]]
                direction1 = np.arctan2(dr1,da1)



                ## extend start point
                ns = tree.query([[c0, s0, r0]], k=min(num_neighbours, min_num_neighbours), return_distance=False)
                ns = np.concatenate(ns)

                direction = np.arctan2(r0 - r[ns], a0 - a[ns])
                diff = 1 - np.cos(direction - direction0)
                ns = ns[(r0 - r[ns] > 0.01) & (diff < (1 - np.cos(limit)))]
                for n in ns: df.loc[df.hit_id == hit_ids[n], 'track_id'] = p

                ## extend end point
                ns = tree.query([[c1, s1, r1]], k=min(num_neighbours, min_num_neighbours), return_distance=False)
                ns = np.concatenate(ns)

                direction = np.arctan2(r[ns] - r1, a[ns] - a1)
                diff = 1 - np.cos(direction - direction1)
                ns = ns[(r[ns] - r1 > 0.01) & (diff < (1 - np.cos(limit)))]
                for n in ns:  df.loc[df.hit_id == hit_ids[n], 'track_id'] = p
        #ytt try
        self.clusters = df["track_id"].copy()

        #print ('\r')
        df = df[['event_id', 'hit_id', 'track_id']]
        return df    

    def make_submission(self, predictor, path):
        """
        :param predictor: function, predictor(hits: pd.DataFrame, cells: pd.DataFrame)->np.array
         takes in hits and cells data frames, return a numpy 1d array of cluster ids
        :param path: file path for submission file
        """
        sub_list = []  # list of predictions by event
        for event_id in self._test_event_id_list:
            event_name = Session.get_event_name(event_id)
            
            hits, cells = load_event(self._parent_dir + self._test_dir + event_name, (Session.HITS, Session.CELLS))
            pred = predictor(hits, cells)  # predicted cluster labels
            sub = pd.DataFrame({"hit_id": hits.hit_id, "track_id": pred})
            sub.insert(0, "event_id", event_id)
            sub_list.append(sub)
        final_submission = pd.concat(sub_list)
        final_submission.to_csv(path, sep=",", header=True, index=False)
    
    def _preprocess(self, hits):
        
        x = hits.x.values
        y = hits.y.values
        z = hits.z.values

        r = np.sqrt(x**2 + y**2 + z**2)
        
        hits['x2'] = x/r
        hits['y2'] = y/r

        r = np.sqrt(x**2 + y**2)
        hits['z2'] = z/r

        ss = StandardScaler()
        X = ss.fit_transform(hits[['x2', 'y2', 'z2']].values)
        for i, rz_scale in enumerate(self.rz_scales):
            X[:,i] = X[:,i] * rz_scale
       
        return X

    def _find_labels(self,params):
        dfhin, dz, ii, stepeps,jj,rtzi,zzi,scanset= params
        dfh = dfhin.copy()
        dfh['z'] = (dfh['z'] + jj) /zzi
        dfh['r'] = np.sqrt(dfh['x'].values**2+dfh['y'].values**2+dfh['z'].values**2)
        dfh['rt'] = np.sqrt(dfh['x'].values**2+dfh['y'].values**2)/rtzi
        dfh['a0'] = np.arctan2(dfh['y'].values,dfh['x'].values)
        dfh['z1'] = dfh['z'].values/dfh['rt'].values
        dfh['z2'] = dfh['z'].values/dfh['r'].values
        dfh['xy'] = dfh['x'].values/dfh['y'].values
        dfh['xr'] = dfh['x'].values/dfh['r'].values
        dfh['yr'] = dfh['y'].values/dfh['r'].values
        #original
        #dfh['a1'] = dfh['a0'].values+dz*dfh['z'].values*np.sign(dfh['z'].values)
        #hough transfer 0522

        dfh['a1'] = dfh['a0'].values - np.arccos(dz*dfh['rt'].values)
        cond=np.where(np.isfinite(dfh['a1'].values))
        dfh['sina1'] = np.zeros(len(dfh))
        dfh['cosa1'] = np.zeros(len(dfh))
        dfh['sina1'].values[cond] = np.sin(dfh['a1'].values[cond])
        dfh['cosa1'].values[cond] = np.cos(dfh['a1'].values[cond])

        #dfh['a1'] = dfh['a0'].values - np.nan_to_num(np.arccos(dz*dfh['rt'].values))
        #dfh['sina1'] = np.sin(dfh['a1'].values)
        #dfh['cosa1'] = np.cos(dfh['a1'].values)
        ss = StandardScaler()

        dfs = ss.fit_transform(dfh[['sina1','cosa1','z1','z2','xr','yr']].values)
        cx = np.array([1.5,1.5,0.73,0.17,0.027,0.027]) #[phi_coef,phi_coef,zdivrt_coef,zdivr_coef,xdivr_coef,ydivr_coef]
        for k in range(6):            
            dfs[:,k] *= cx[k]

        if(scanset ==2):
          clusters=DBSCAN(eps=0.005+ii*10*0.000001,min_samples=1,metric="euclidean",n_jobs=8).fit(dfs).labels_ 
        else:
          clusters=DBSCAN(eps=0.005+ii*10*0.000001,min_samples=1,metric="euclidean",n_jobs=8).fit(dfs).labels_ 

        result = self._add_count(clusters + 1,dfh,scanset)
        return result

    def _init(self,dfhin,stage=0,newstart=1):

        start_time = timeit.default_timer()
        print(type(dfhin))
        print(dfhin)
        dfh = dfhin.copy() 
        volume_id = dfh['volume_id'].values.astype(np.float32)
        layer_id = dfh['layer_id'].values.astype(np.float32)
        module_id = dfh['module_id'].values.astype(np.float32)

        #hough
        dz0=0
        mm = 1
        init=0
        print("Init")

        if(stage==0):
            scanuplim = 360
            scanlowlim = 0
            stepeps = 0.0000005
            stepdz = 0.000005            
        elif(stage==1):
            scanuplim = 360
            scanlowlim = 0        
            stepeps = 0.0000005
            stepdz = 0.000005 
        elif(stage==2):
            scanuplim = 360
            scanlowlim = 0
            stepeps = 0.0000005 
            stepdz = 0.000005   
        elif(stage==3):
            scanuplim = 360
            scanlowlim = 0
            stepeps = 0.0000005 
            stepdz = 0.000005    
        elif(stage==4):
            scanuplim = 360
            scanlowlim = 0
            stepeps = 0.0000005 
            stepdz = 0.00001     
        elif(stage==5):
            scanuplim = 300
            scanlowlim = 0
            stepeps = 0.0000005 
            stepdz = 0.00001 
        else:
            scanuplim = 300
            scanlowlim = 0
            stepeps = 0.0000005 
            stepdz = 0.00001                                                

        params = []
        EPS=1e-12
        
        zshift_range = [11,-11,7,-7,4,-4,0] #good #627

        rtzip = [2,1] #good
        
        zzzip = [1]
        
        print("Slicing scanning parameters....")
        for rtzi in rtzip:
          for zzi in zzzip:
            for jj in zshift_range: 
              #print("jj")
              #print(jj)
              for ii in np.arange(scanlowlim,scanuplim):    
                mm = mm*(-1)
                dz = mm*(dz0+ii*stepdz)
                params.append((dfh, dz,ii,stepeps,jj,rtzi,zzi,stage ))

        print("Done!")
        #new test
        n=8
        print("Initializing parallel pool: {} processes".format(n))
        pool = Pool(processes=n)
        print("Fitting...")

        results = pool.map(self._find_labels, params)
        pool.close()

        labels, counts = results[0]
        print("Results:")
        print(len(labels))
        print(len(results))
        print(len(counts))
        #ytt for all data second scan
        if stage !=0 and newstart ==0: #use last data before
                print("st0 , newstart0")
                dfh['s1'] = self.clusters
                dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
                labels = dfh['s1']
                counts = dfh['N1'] 
        else:#USE first data of NEW trackid
            print("new track , use first index")
            dfh['s1'] = labels
            counts =dfh.groupby('s1')['s1'].transform('count')
            dfh['N1'] = counts

        print("count size")    
        print(len(counts))    
        for i in range(0, len(results)):
            l, c = results[i]
            dfh['s2'] = l
            dfh['N1'] = c
            dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
            dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
            maxs1 = dfh['s1'].max()
            cond = np.where((dfh['N2'].values>dfh['N1'].values) & (dfh['N2'].values<25)) # 
            s1 = dfh['s1'].values 
            s1[cond] = dfh['s2'].values[cond]+maxs1 # 

        print('Time spent:', timeit.default_timer() - start_time)
        return dfh['s1'].values  

    def _eliminate_outliers(self,dfh,labels,M,stage=0):
        norms=np.zeros((len(labels)),np.float32)
        indices=np.zeros((len(labels)),np.float32)
        
        volume_id = dfh['volume_id'].values.astype(np.float32)
        layer_id = dfh['layer_id'].values.astype(np.float32)
        module_id = dfh['module_id'].values.astype(np.float32)
        hitsx = dfh[["x","y","z"]]
        print("eliminate")
        print("check threshold")
        if(stage==0):
            mergelen_threshold = 12
            sensor_diff_threshold = 0
            lowlen_threshold = 12
        elif(stage==1):
            mergelen_threshold = 9
            sensor_diff_threshold = 1
            lowlen_threshold = 9   
        elif(stage==2):
            mergelen_threshold = 9
            sensor_diff_threshold = 5
            lowlen_threshold = 6                     
        elif(stage==3):
            mergelen_threshold = 6
            sensor_diff_threshold = 2
            lowlen_threshold = 6
        elif(stage==4):
            mergelen_threshold = 4
            sensor_diff_threshold = 5
            lowlen_threshold = 4

        for i, cluster in tqdm(enumerate(labels),total=len(labels)):
            if cluster == 0:
                continue
            #for all pair , count there norm    
            index = np.argwhere(self.clusters==cluster)
            index = np.reshape(index,(index.shape[0]))
            #print(type(index))
            indices[i] = len(index)
            x = M[index]

            # check if follow helix line
            norms[i] = self._test_quadric(x)

            if(len(index)>=mergelen_threshold):

                currentdf = dfh.iloc[index]
                current_module =module_id[index] 
                current_layer =layer_id[index] 

                module_id0 = current_module[:-1]
                module_id1 = current_module[1: ]
                ld = module_id1-module_id0
                #the same module id is not allowed
                #print(ld)
                layer_id0 = current_layer[:-1]
                layer_id1 = current_layer[1: ]
                ld2 = layer_id1-layer_id0
                #print(ld2)
                #ld = np.where(ld==0 and ld2==0)
                #print(ld)
                checkindex = np.where(ld==0)
                checkindex2 = np.where(ld2==0)
                checkindex =np.intersect1d(checkindex ,checkindex2)#

                #print(checkindex)
                #print(len(checkindex))

                if(len(checkindex)>sensor_diff_threshold):
                    self.clusters[self.clusters==cluster]=0 
                    continue

        print(norms)
        threshold1 = np.percentile(norms,90)*5  # +-10%   *5
        #threshold1 = 1
        print(threshold1)
        threshold2 = 20 #length >25


        threshold3 = lowlen_threshold


        #threshold3 = 4  #length <6  as i know the min length is 4        
        for i, cluster in enumerate(labels):
            if norms[i] > threshold1 or indices[i] > threshold2 or indices[i] < threshold3:
            #if norms[i] > threshold1 :
               #if(indices[i] !=1): # I guess its noise
                self.clusters[self.clusters==cluster]=0 

    def _add_count(self,l,dfh,stage):

        if(stage==0):
            mergelen_threshold = 12
            sensor_diff_threshold = 0
            lowlen_threshold = 12
        elif(stage==1):
            mergelen_threshold = 9
            sensor_diff_threshold = 1
            lowlen_threshold = 12   
        elif(stage==2):
            mergelen_threshold = 9
            sensor_diff_threshold = 5
            lowlen_threshold = 6                     
        elif(stage==3):
            mergelen_threshold = 6
            sensor_diff_threshold = 3
            lowlen_threshold = 6
        elif(stage==4): # free run
            mergelen_threshold = 9
            sensor_diff_threshold = 2
            lowlen_threshold = 4
        elif(stage==5): # free run
            mergelen_threshold = 9
            sensor_diff_threshold = 2
            lowlen_threshold = 4
        else:            
            unique, reverse, count = np.unique(l, return_counts=True, return_inverse=True)
            c = count[reverse]
            #clean count by label
            c[np.where(l == 0)] = 0
            c[np.where(c > 20)] = 0
            return (l, c)  

        #labels = np.unique(l)
        unique, reverse, count = np.unique(l, return_counts=True, return_inverse=True)
        c = count[reverse]
        #clean count by label
        c[np.where(l == 0)] = 0
        c[np.where(c > 20)] = 0

 
        #not alloweed non codradic
        # l2_norm = l[np.where(c >= 6)]
        # labels_old = np.unique(l2_norm)
        # norms=np.zeros((len(labels_old)),np.float32)
        # indices=np.zeros((len(labels_old)),np.float32)
        # print("x:M")
        # M = self.X
        # for i, cluster in tqdm(enumerate(labels_old),total=len(labels_old)):
        #     if cluster == 0:
        #         continue
        #     index = np.argwhere(l==cluster)
        #     index = np.reshape(index,(index.shape[0]))    
        #     x = M[index]
        #     norms[i] = self._test_quadric(x)

        #     print(norms[i])
        #     if np.log10(norms[i]) >0:
        #         print("kill norm !!!!")
        #         l[l==cluster]=0 
        #         continue  



        l2 = l[np.where(c >= mergelen_threshold)]
        labels = np.unique(l2)
        #print(len(labels))
        #print(labels)
        #print(l2)
      
        indices=np.zeros((len(labels)),np.float32)
        #M = self.X
   
        for i, cluster in tqdm(enumerate(labels),total=len(labels),disable=True):
            if cluster == 0:
                continue
            #for all pair , count there norm    
            index = np.argwhere(l==cluster)
            index = np.reshape(index,(index.shape[0]))
            #print(type(index))
            indices[i] = len(index)
                 

            # check if follow helix line

            #if(len(index)>=mergelen_threshold):
            if(len(index)>=mergelen_threshold):


                #print(x)
                #print(type(index))
                #print(index)
                #truthdf = truth.iloc[index]
                currentdf = dfh.iloc[index]
                #print(truthdf)
                #print(currentdf)
                #print(norms[i])

                volume_id = dfh['volume_id'].values.astype(np.float32)
                layer_id = dfh['layer_id'].values.astype(np.float32)
                module_id = dfh['module_id'].values.astype(np.float32)
                current_module =module_id[index] 
                current_layer =layer_id[index] 
                module_id0 = current_module[:-1]
                module_id1 = current_module[1: ]
                ld = module_id1-module_id0
                #the same module id is not allowed
                #print(ld)
                layer_id0 = current_layer[:-1]
                layer_id1 = current_layer[1: ]
                ld2 = layer_id1-layer_id0
                #print(ld2)
                #ld = np.where(ld==0 and ld2==0)
                #print(ld)
                checkindex = np.where(ld==0)
                checkindex2 = np.where(ld2==0)
                checkindex =np.intersect1d(checkindex ,checkindex2)#

                #print(checkindex)
                #print(len(checkindex))

                if(len(checkindex)>sensor_diff_threshold):
                    #print("kill!!!!")
                    #print(truthdf)
                    l[l==cluster]=0 
                    continue
                #if(len(checkindex)>sensor_diff_threshold):
                #    self.clusters[self.clusters==cluster]=0 
                #    continue


 
        #test
        #threshold1 = np.percentile(norms,90)*5  # +-10%   *5
        #threshold2 = 25 #length >25
        #threshold3 = 6  #length <6  as i know the min length is 4
        # print(norms)
        # threshold1 = np.percentile(norms,90)*5  # +-10%   *5
        # #threshold1 = 1
        # print(threshold1)
        # threshold2 = 25 #length >25
        # threshold3 = lowlen_threshold
        # #threshold3 = 4  #length <6  as i know the min length is 4        
        # for i, cluster in enumerate(labels):
        #     if norms[i] > threshold1 or indices[i] > threshold2 or indices[i] < threshold3:
        #     #if norms[i] > threshold1 :
        #        #if(indices[i] !=1): # I guess its noise
        #         self.clusters[self.clusters==cluster]=0




        unique, reverse, count = np.unique(l, return_counts=True, return_inverse=True)
        c = count[reverse]
        #clean count by label
        c[np.where(l == 0)] = 0
        c[np.where(c > 20)] = 0
        return (l, c)

    def predict(self, hits):    
        dataset_submissions=[]

        hits = hits.assign(rrr  = np.sqrt( hits.x**2 + hits.y**2))

        print("Preprocessing.....") 
        X = self._preprocess(hits)
        self.X = X
        print("Done!")
        
        print("Start Clustering.....") 
        self.clusters = self._init(hits,stage=0) #90
        print("Clustering Finished!") 


        print("Eliminating outliers.....") 
        labels = np.unique(self.clusters)
        self._eliminate_outliers(hits,labels,X,stage=0)
        print("Done!") 
        
        n_labels = 0
        while n_labels < len(labels):
            n_labels = len(labels) 
            print("len label 1")
            print(len(labels))            
            #total hit mapping           
            max_len = np.max(self.clusters)
            #get all eliminator and predict again
            mask = self.clusters == 0    
            print("outlier ")
            print(len(mask))        
            self.clusters[mask] = self._init(hits[mask],stage=1)+max_len          
            print("len label 2")
            print(len(labels))

        #stage 1-2
        labels = np.unique(self.clusters)
        self._eliminate_outliers(hits,labels,X,stage=1)
        
        one_submission = create_one_event_submission(event_id, hits, self.clusters)
        dataset_submissions.append(one_submission)
        score = score_event(truth, one_submission)
        print("Score after eliminate stage 1 event : %.8f" % ( score))  

        max_len = np.max(self.clusters)
        mask = self.clusters == 0 
        print("outlier ")
        print(len(mask))         
        self.clusters[mask] = self._init(hits[mask],stage=3)+max_len 

        one_submission = create_one_event_submission(event_id, hits, self.clusters)
        dataset_submissions.append(one_submission)
        score = score_event(truth, one_submission)
        print("Score after stage 1-2 event : %.8f" % ( score))

        return self.clusters   


def cossimilar(X,Y):
    lx = np.sqrt(np.inner(X,X))
    ly = np.sqrt(np.inner(Y,Y))
    np.inner(X, Y)
    cossimilar = np.inner(X, Y) / (lx*ly)
    return cossimilar

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(np.int64)
    return submission

#https://www.kaggle.com/cpmpml/a-faster-python-scoring-function
def cpmp_fast_score(truth, submission):

    truth = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    df    = truth.groupby(['track_id', 'particle_id']).hit_id.count().to_frame('count_both').reset_index()
    truth = truth.merge(df, how='left', on=['track_id', 'particle_id'])

    df1   = df.groupby(['particle_id']).count_both.sum().to_frame('count_particle').reset_index()
    truth = truth.merge(df1, how='left', on='particle_id')
    df1   = df.groupby(['track_id']).count_both.sum().to_frame('count_track').reset_index()
    truth = truth.merge(df1, how='left', on='track_id')
    truth.count_both *= 2

    score = truth[(truth.count_both > truth.count_particle) & (truth.count_both > truth.count_track)].weight.sum()
    results = truth


    return score, results

if __name__ == "__main__":
    #s1 = Session(parent_dir="/mydisk/TrackML-Data/tripletnet/")
    #event_names, event_loaders = s1.remove_train_events(4, content=[s1.HITS, s1.TRUTH], randomness=True)
    i = 0