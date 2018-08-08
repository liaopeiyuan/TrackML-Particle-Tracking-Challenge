import os
import timeit
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from trackml.dataset import load_event
from trackml.score import score_event

from finalAtom.utils.create_submission import create_one_event_submission


class Clusterer(object):
    event_id = ""
    event_prefix = ""

    path_to_train = ""

    def __init__(self, event_id, event_prefix, path_to_train, rz_scales=[0.65, 0.965, 1.528]):
        self.event_id = event_id
        self.event_prefix = event_prefix
        self.path_to_train = path_to_train
        self.rz_scales = rz_scales
    
    def _init_merge(self, dfhin, stage=0, newstart=1):
        start_time = timeit.default_timer()
        print(type(dfhin))
        print(dfhin)
        dfh = dfhin.copy()
        # ytt z xhift

        volume_id = dfh['volume_id'].values.astype(np.float32)
        layer_id = dfh['layer_id'].values.astype(np.float32)
        module_id = dfh['module_id'].values.astype(np.float32)

        dz0 = 0
        mm = 1
        init = 0
        print("gogola")

        if stage == 0:
            scanuplim = 360
            scanlowlim = 0
            stepeps = 0.0000005
            stepdz = 0.000005
        elif stage == 1:
            scanuplim = 360
            scanlowlim = 0
            stepeps = 0.0000005
            stepdz = 0.000005
        elif stage == 2:
            scanuplim = 360
            scanlowlim = 0
            stepeps = 0.0000005
            stepdz = 0.000005
        elif stage == 3:
            scanuplim = 360
            scanlowlim = 0
            stepeps = 0.0000005
            stepdz = 0.000005
        elif stage == 4:
            scanuplim = 360
            scanlowlim = 0
            stepeps = 0.0000005
            stepdz = 0.00001
        elif stage == 5:
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
        EPS = 1e-12
        zshift_range = [11, -11, 7, -7, 4, -4, 0]  # good #627
        rtzip = [2, 1]  # good
        zzzip = [1]

        for rtzi in rtzip:
            # for ti in temp:
            for zzi in zzzip:
                for jj in zshift_range:
                    # for jj in np.arange(-7.501,7.5,3):
                    print("jj")
                    print(jj)
                    for ii in tqdm(np.arange(scanlowlim, scanuplim)):
                        mm = mm * (-1)
                        dz = mm * (dz0 + ii * stepdz)
                        # dz = dz0+ii*stepdz
                        params.append((dfh, dz, ii, stepeps, jj, rtzi, zzi, stage))
                        # params.append((dfh,-dz,ii,stepeps,jj))

        pool = Pool(processes=8)
        results = pool.map(self._find_labels, params)
        pool.close()

        labels, counts = results[0]
        print("all result")
        print(len(labels))
        print(len(results))
        print(len(counts))

        print('time spent:', timeit.default_timer() - start_time)
        return results

    def _init(self, dfhin, stage=0, newstart=1):

        start_time = timeit.default_timer()
        print(type(dfhin))
        print(dfhin)
        dfh = dfhin.copy()
        # ytt z xhift

        volume_id = dfh['volume_id'].values.astype(np.float32)
        layer_id = dfh['layer_id'].values.astype(np.float32)
        module_id = dfh['module_id'].values.astype(np.float32)

        dz0 = 0
        mm = 1
        init = 0
        print("gogola")

        if stage == 0:
            scanuplim = 360
            scanlowlim = 0
            stepeps = 0.0000005
            stepdz = 0.000005
        elif stage == 1:
            scanuplim = 360
            scanlowlim = 0
            stepeps = 0.0000005
            stepdz = 0.000005
        elif stage == 2:
            scanuplim = 360
            scanlowlim = 0
            stepeps = 0.0000005
            stepdz = 0.000005
        elif stage == 3:
            scanuplim = 360
            scanlowlim = 0
            stepeps = 0.0000005
            stepdz = 0.000005
        elif stage == 4:
            scanuplim = 360
            scanlowlim = 0
            stepeps = 0.0000005
            stepdz = 0.00001
        elif stage == 5:
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
        EPS = 1e-12
        zshift_range = [11, -11, 7, -7, 4, -4, 0]  # good #627
        rtzip = [2, 1]  # good
        zzzip = [1]

        for rtzi in rtzip:
            # for ti in temp:
            for zzi in zzzip:
                for jj in zshift_range:
                    # for jj in np.arange(-7.501,7.5,3):
                    print("jj")
                    print(jj)
                    for ii in tqdm(np.arange(scanlowlim, scanuplim)):
                        mm = mm * (-1)
                        dz = mm * (dz0 + ii * stepdz)
                        # dz = dz0+ii*stepdz
                        params.append((dfh, dz, ii, stepeps, jj, rtzi, zzi, stage))
                        # params.append((dfh,-dz,ii,stepeps,jj))

        pool = Pool(processes=8)
        results = pool.map(self._find_labels, params)
        pool.close()

        labels, counts = results[0]
        print("all result")
        print(len(labels))
        print(len(results))
        print(len(counts))
        # ytt for all data second scan
        if stage != 0 and newstart == 0:  # use last data before
            print("st0 , newstart0")
            dfh['s1'] = self.clusters
            dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
            labels = dfh['s1']
            counts = dfh['N1']
        else:  # USE first data of NEW trackid
            print("new track , use first index")
            dfh['s1'] = labels
            counts = dfh.groupby('s1')['s1'].transform('count')
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
            cond = np.where((dfh['N2'].values > dfh['N1'].values) & (dfh['N2'].values < 25))  #
            s1 = dfh['s1'].values
            s1[cond] = dfh['s2'].values[cond] + maxs1  #

        print('time spent:', timeit.default_timer() - start_time)
        return dfh['s1'].values

    def _init2(self, dfh):
        print(type(dfh))
        print(dfh)
        dfh = dfh.copy()
        # rr = np.sqrt(dfh['x'].values**2+dfh['y'].values**2+dfh['z'].values**2)

        dfh['r'] = np.sqrt(dfh['x'].values ** 2 + dfh['y'].values ** 2 + dfh['z'].values ** 2)
        dfh['rt'] = np.sqrt(dfh['x'].values ** 2 + dfh['y'].values ** 2)
        dfh['a0'] = np.arctan2(dfh['y'].values, dfh['x'].values)
        dfh['z1'] = dfh['z'].values / dfh['rt'].values
        dfh['x2'] = 1 / dfh['z1'].values
        # set all z ->x
        dz0 = -0.00070
        stepdz = 0.00001
        stepeps = 0.000005
        # stepeps = 0.000001
        mm = 1
        print("gogola2")
        for ii in tqdm(range(100)):
            mm = mm * (-1)
            dz = mm * (dz0 + ii * stepdz)
            dfh['a1'] = dfh['a0'].values + dz * dfh['z'].values * np.sign(dfh['z'].values)
            dfh['sina1'] = np.sin(dfh['a1'].values)
            dfh['cosa1'] = np.cos(dfh['a1'].values)
            dfh['x1'] = dfh['a1'].values / dfh['z1'].values
            ss = StandardScaler()
            dfs = ss.fit_transform(dfh[['sina1', 'cosa1', 'z1', 'x1', 'x2']].values)

            cx = np.array([1, 1, 0.75, 0.5, 0.5])
            # cx = np.array([1, 1, 0.75])
            for k in range(5):
                # for k in range(3):
                dfs[:, k] *= cx[k]
            clusters = DBSCAN(eps=0.0035 + ii * stepeps, min_samples=1, metric='euclidean', n_jobs=4).fit(dfs).labels_

            if ii == 0:
                dfh['s1'] = clusters  # index +1 ,0 for empty
                # get link long fro each item
                dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
                print(dfh['N1'].head())

            else:
                dfh['s2'] = clusters
                # get link long for each item
                dfh['N2'] = dfh.groupby('s2')['s2'].transform('count')
                # get label max value for s1
                maxs1 = dfh['s1'].max()
                # get all hit label item that link long > previous
                cond = np.where((dfh['N2'].values > dfh['N1'].values) & (dfh['N2'].values < 19))
                # replace shorter one with longer path
                s1 = dfh['s1'].values
                # replace longer path with new trackid
                s1[cond] = dfh['s2'].values[cond] + maxs1
                # update final path
                dfh['s1'] = s1
                dfh['s1'] = dfh['s1'].astype('int64')
                # update link long
                dfh['N1'] = dfh.groupby('s1')['s1'].transform('count')
        # heng'code

        return dfh['s1'].values

    def _preprocess(self, hits):

        x = hits.x.values
        y = hits.y.values
        z = hits.z.values

        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        hits['x2'] = x / r
        hits['y2'] = y / r

        r = np.sqrt(x ** 2 + y ** 2)
        hits['z2'] = z / r

        ss = StandardScaler()
        X = ss.fit_transform(hits[['x2', 'y2', 'z2']].values)

        for i, rz_scale in enumerate(self.rz_scales):
            X[:, i] = X[:, i] * rz_scale

        return X

    def _find_labels(self, params):
        dfhin, dz, ii, stepeps, jj, rtzi, zzi, scanset = params
        dfh = dfhin.copy()
        # dz = mm*(dz0+ii*stepdz)
        dfh['z'] = (dfh['z'] + jj) / zzi
        dfh['r'] = np.sqrt(dfh['x'].values ** 2 + dfh['y'].values ** 2 + dfh['z'].values ** 2)
        dfh['rt'] = np.sqrt(dfh['x'].values ** 2 + dfh['y'].values ** 2) / rtzi
        dfh['a0'] = np.arctan2(dfh['y'].values, dfh['x'].values)
        dfh['z1'] = dfh['z'].values / dfh['rt'].values
        dfh['z2'] = dfh['z'].values / dfh['r'].values
        dfh['xy'] = dfh['x'].values / dfh['y'].values
        dfh['xr'] = dfh['x'].values / dfh['r'].values
        dfh['yr'] = dfh['y'].values / dfh['r'].values
        # original
        # dfh['a1'] = dfh['a0'].values+dz*dfh['z'].values*np.sign(dfh['z'].values)
        # hough transfer 0522
        dfh['a1'] = dfh['a0'].values - np.nan_to_num(np.arccos(dz * dfh['rt'].values))
        dfh['sina1'] = np.sin(dfh['a1'].values)
        dfh['cosa1'] = np.cos(dfh['a1'].values)
        ss = StandardScaler()

        # dfs = ss.fit_transform(dfh[['sina1','cosa1','z1','z2','xy','xr','yr']].values)
        # original
        # cx = np.array([1, 1,0.4, 0.4,0.005,0.005,0.005])
        # for k in range(7):
        dfs = ss.fit_transform(dfh[['sina1', 'cosa1', 'z1', 'z2', 'xr', 'yr']].values)
        cx = np.array([1.5, 1.5, 0.73, 0.17, 0.027, 0.027])  # [phi_coef,phi_coef,zdivrt_coef,zdivr_coef,xdivr_coef,ydivr_coef]
        for k in range(6):
            dfs[:, k] *= cx[k]

            # if(stage==0):
            #    clusters=DBSCAN(eps=0.0035+ii*stepeps,min_samples=1,metric="chebyshev",n_jobs=4).fit(dfs).labels_
            # else:
        # clusters=DBSCAN(eps=0.0035+ii*stepeps,min_samples=1,metric="euclidean",n_jobs=8).fit(dfs).labels_\
        # hough
        # xxxxclusters=DBSCAN(eps=0.005+ii*10**(-5),min_samples=1,metric="euclidean",n_jobs=8).fit(dfs).labels_
        if scanset == 2:
            clusters = DBSCAN(eps=0.005 + ii * 10 * 0.000001, min_samples=1, metric="euclidean", n_jobs=8).fit(
                dfs).labels_
        else:
            clusters = DBSCAN(eps=0.005 + ii * 10 * 0.000001, min_samples=1, metric="euclidean", n_jobs=8).fit(
                dfs).labels_
            # clusters=DBSCAN(eps=0.005,min_samples=1,metric="euclidean",n_jobs=8).fit(dfs).labels_

        # self._add_count(l,dfh,stage)
        result = self._add_count(clusters + 1, dfh, scanset)
        return result

    def _add_count_default(self, l, dfh):
        unique, reverse, count = np.unique(l, return_counts=True, return_inverse=True)
        c = count[reverse]
        # clean count by label
        c[np.where(l == 0)] = 0
        c[np.where(c > 20)] = 0
        return l, c  # labels = np.unique(l)

    def _add_count(self, l, dfh, stage):

        if stage == 0:
            mergelen_threshold = 12
            sensor_diff_threshold = 0
            lowlen_threshold = 12
        elif stage == 1:
            mergelen_threshold = 9
            sensor_diff_threshold = 1
            lowlen_threshold = 12
        elif stage == 2:
            mergelen_threshold = 9
            sensor_diff_threshold = 5
            lowlen_threshold = 6
        elif stage == 3:
            mergelen_threshold = 6
            sensor_diff_threshold = 3
            lowlen_threshold = 6
        elif stage == 4:  # free run
            mergelen_threshold = 9
            sensor_diff_threshold = 2
            lowlen_threshold = 4
        elif stage == 5:  # free run
            mergelen_threshold = 9
            sensor_diff_threshold = 2
            lowlen_threshold = 4
        else:
            unique, reverse, count = np.unique(l, return_counts=True, return_inverse=True)
            c = count[reverse]
            # clean count by label
            c[np.where(l == 0)] = 0
            c[np.where(c > 20)] = 0
            return l, c

            # labels = np.unique(l)
        unique, reverse, count = np.unique(l, return_counts=True, return_inverse=True)
        c = count[reverse]
        # clean count by label
        c[np.where(l == 0)] = 0
        c[np.where(c > 20)] = 0

        l2 = l[np.where(c >= mergelen_threshold)]
        labels = np.unique(l2)

        indices = np.zeros((len(labels)), np.float32)

        for i, cluster in tqdm(enumerate(labels), total=len(labels)):
            if cluster == 0:
                continue
            # for all pair , count there norm
            index = np.argwhere(l == cluster)
            index = np.reshape(index, (index.shape[0]))
            # print(type(index))
            indices[i] = len(index)

            # check if follow helix line

            # if(len(index)>=mergelen_threshold):
            if len(index) >= mergelen_threshold:

                currentdf = dfh.iloc[index]

                volume_id = dfh['volume_id'].values.astype(np.float32)
                layer_id = dfh['layer_id'].values.astype(np.float32)
                module_id = dfh['module_id'].values.astype(np.float32)
                current_module = module_id[index]
                current_layer = layer_id[index]
                module_id0 = current_module[:-1]
                module_id1 = current_module[1:]
                ld = module_id1 - module_id0

                layer_id0 = current_layer[:-1]
                layer_id1 = current_layer[1:]
                ld2 = layer_id1 - layer_id0

                checkindex = np.where(ld == 0)
                checkindex2 = np.where(ld2 == 0)
                checkindex = np.intersect1d(checkindex, checkindex2)  #

                # print(checkindex)
                # print(len(checkindex))

                if len(checkindex) > sensor_diff_threshold:
                    # print("kill!!!!")
                    # print(truthdf)
                    l[l == cluster] = 0
                    continue

        unique, reverse, count = np.unique(l, return_counts=True, return_inverse=True)
        c = count[reverse]
        # clean count by label
        c[np.where(l == 0)] = 0
        c[np.where(c > 20)] = 0
        return l, c

    def _extend(self, submission, hits, limit=0.04, num_neighbours=18):
        df = submission.merge(hits, on=['hit_id'], how='left')
        df = df.assign(d=np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2))
        df = df.assign(r=np.sqrt(df.x ** 2 + df.y ** 2))
        df = df.assign(arctan2=np.arctan2(df.z, df.r))

        # for angle in range( -90,90,1):
        # ytt
        for angle in np.arange(-90, 90, 0.5):

            print('\r %f' % angle, end='', flush=True)
            # df1 = df.loc[(df.arctan2>(angle-0.5)/180*np.pi) & (df.arctan2<(angle+0.5)/180*np.pi)] #bad
            df1 = df.loc[(df.arctan2 > (angle - 1.5) / 180 * np.pi) & (df.arctan2 < (angle + 1.5) / 180 * np.pi)]

            min_num_neighbours = len(df1)
            if min_num_neighbours < 3: continue

            hit_ids = df1.hit_id.values
            x, y, z = df1[['x', 'y', 'z']].values.T
            r = (x ** 2 + y ** 2) ** 0.5
            r = r / 1000
            a = np.arctan2(y, x)
            c = np.cos(a)
            s = np.sin(a)
            # tree = KDTree(np.column_stack([a,r]), metric='euclidean')
            tree = KDTree(np.column_stack([c, s, r]), metric='euclidean')

            track_ids = list(df1.track_id.unique())
            num_track_ids = len(track_ids)
            # min_length=3
            # ytt
            min_length = 3

            for i in range(num_track_ids):
                p = track_ids[i]
                if p == 0: continue

                idx = np.where(df1.track_id == p)[0]
                if len(idx) < min_length: continue

                if angle > 0:
                    idx = idx[np.argsort(z[idx])]
                else:
                    idx = idx[np.argsort(-z[idx])]

                ## start and end points  ##
                idx0, idx1 = idx[0], idx[-1]
                a0 = a[idx0]
                a1 = a[idx1]
                r0 = r[idx0]
                r1 = r[idx1]
                c0 = c[idx0]
                c1 = c[idx1]
                s0 = s[idx0]
                s1 = s[idx1]

                da0 = a[idx[1]] - a[idx[0]]  # direction
                dr0 = r[idx[1]] - r[idx[0]]
                direction0 = np.arctan2(dr0, da0)

                da1 = a[idx[-1]] - a[idx[-2]]
                dr1 = r[idx[-1]] - r[idx[-2]]
                direction1 = np.arctan2(dr1, da1)

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
        # ytt try
        self.clusters = df["track_id"].copy()

        # print ('\r')
        df = df[['event_id', 'hit_id', 'track_id']]
        return df

    def _eliminate_outliers(self, dfh, labels, M, stage=0):
        norms = np.zeros((len(labels)), np.float32)
        indices = np.zeros((len(labels)), np.float32)

        volume_id = dfh['volume_id'].values.astype(np.float32)
        layer_id = dfh['layer_id'].values.astype(np.float32)
        module_id = dfh['module_id'].values.astype(np.float32)
        hitsx = dfh[["x", "y", "z"]]
        print("eliminate")
        print("check threshold")
        if stage == 0:
            mergelen_threshold = 12
            sensor_diff_threshold = 0
            lowlen_threshold = 12
        elif stage == 1:
            mergelen_threshold = 9
            sensor_diff_threshold = 1
            lowlen_threshold = 9
        elif stage == 2:
            mergelen_threshold = 9
            sensor_diff_threshold = 5
            lowlen_threshold = 6
        elif stage == 3:
            mergelen_threshold = 6
            sensor_diff_threshold = 2
            lowlen_threshold = 6
        elif stage == 4:
            mergelen_threshold = 4
            sensor_diff_threshold = 5
            lowlen_threshold = 4

        for i, cluster in tqdm(enumerate(labels), total=len(labels)):
            if cluster == 0:
                continue
            # for all pair , count there norm
            index = np.argwhere(self.clusters == cluster)
            index = np.reshape(index, (index.shape[0]))
            # print(type(index))
            indices[i] = len(index)
            x = M[index]

            # check if follow helix line
            norms[i] = self._test_quadric(x)

            if len(index) >= mergelen_threshold:

                currentdf = dfh.iloc[index]
                current_module = module_id[index]
                current_layer = layer_id[index]
                module_id0 = current_module[:-1]
                module_id1 = current_module[1:]
                ld = module_id1 - module_id0
                # the same module id is not allowed
                # print(ld)
                layer_id0 = current_layer[:-1]
                layer_id1 = current_layer[1:]
                ld2 = layer_id1 - layer_id0
                # print(ld2)
                # ld = np.where(ld==0 and ld2==0)
                # print(ld)
                checkindex = np.where(ld == 0)
                checkindex2 = np.where(ld2 == 0)
                checkindex = np.intersect1d(checkindex, checkindex2)  #

                # print(checkindex)
                # print(len(checkindex))

                if len(checkindex) > sensor_diff_threshold:
                    self.clusters[self.clusters == cluster] = 0
                    continue

        print(norms)
        threshold1 = np.percentile(norms, 90) * 5  # +-10%   *5
        # threshold1 = 1
        print(threshold1)
        threshold2 = 20  # length >25

        threshold3 = lowlen_threshold

        # threshold3 = 4  #length <6  as i know the min length is 4
        for i, cluster in enumerate(labels):
            if norms[i] > threshold1 or indices[i] > threshold2 or indices[i] < threshold3:
                # if norms[i] > threshold1 :
                # if(indices[i] !=1): # I guess its noise
                self.clusters[self.clusters == cluster] = 0

    def _test_quadric(self, x):
        if x.size == 0 or len(x.shape) < 2:
            return 0
        Z = np.zeros((x.shape[0], 10), np.float32)
        Z[:, 0] = x[:, 0] ** 2
        Z[:, 1] = 2 * x[:, 0] * x[:, 1]
        Z[:, 2] = 2 * x[:, 0] * x[:, 2]
        Z[:, 3] = 2 * x[:, 0]
        Z[:, 4] = x[:, 1] ** 2
        Z[:, 5] = 2 * x[:, 1] * x[:, 2]
        Z[:, 6] = 2 * x[:, 1]
        Z[:, 7] = x[:, 2] ** 2
        Z[:, 8] = 2 * x[:, 2]
        Z[:, 9] = 1
        v, s, t = np.linalg.svd(Z, full_matrices=False)
        smallest_index = np.argmin(np.array(s))
        T = np.array(t)
        T = T[smallest_index, :]
        norm = np.linalg.norm(np.dot(Z, T), ord=2) ** 2
        return norm

    def predict(self, hits):

        dataset_submissions = []

        hits, cells, particles, truth = load_event(os.path.join(self.path_to_train, self.event_prefix))

        hits = hits.assign(rrr=np.sqrt(hits.x ** 2 + hits.y ** 2))
        X = self._preprocess(hits)
        self.X = X
        self.clusters = self._init(hits, stage=0)  # 90

        one_submission = create_one_event_submission(self.event_id, hits, self.clusters)
        dataset_submissions.append(one_submission)

        one_submission = create_one_event_submission(self.event_id, hits, self.clusters)
        dataset_submissions.append(one_submission)

        # ytt remove for test
        labels = np.unique(self.clusters)
        self._eliminate_outliers(hits, labels, X, stage=0)

        # count again
        one_submission = create_one_event_submission(self.event_id, hits, self.clusters)
        # for i in range(7): one_submission = self._extend(one_submission, hits)
        dataset_submissions.append(one_submission)

        n_labels = 0
        while n_labels < len(labels):
            n_labels = len(labels)
            print("len label 1")
            print(len(labels))
            # total hit mapping
            max_len = np.max(self.clusters)
            # get all eliminator and predict again
            mask = self.clusters == 0
            print("outlier ")
            print(len(mask))
            # self.clusters[mask] = cl.fit_predict(X[mask])+max_len
            # self.clusters[mask] = self._init2(hits[mask])+max_len
            self.clusters[mask] = self._init(hits[mask], stage=1) + max_len
            # labels = np.unique(self.clusters)
            print("len label 2")
            print(len(labels))

        one_submission = create_one_event_submission(self.event_id, hits, self.clusters)
        # one_submission= self._extend(one_submission,hits)
        dataset_submissions.append(one_submission)
        # score = score_event(truth, one_submission)
        # print("Score after stage1 event : %.8f" % ( score))

        # stage 1-2
        labels = np.unique(self.clusters)
        self._eliminate_outliers(hits, labels, X, stage=1)

        one_submission = create_one_event_submission(self.event_id, hits, self.clusters)
        dataset_submissions.append(one_submission)
        score = score_event(truth, one_submission)
        print("Score after eliminate stage1 event : %.8f" % score)

        max_len = np.max(self.clusters)
        mask = self.clusters == 0
        print("outlier ")
        print(len(mask))
        self.clusters[mask] = self._init(hits[mask], stage=3) + max_len

        one_submission = create_one_event_submission(self.event_id, hits, self.clusters)
        # one_submission= self._extend(one_submission,hits)
        dataset_submissions.append(one_submission)
        score = score_event(truth, one_submission)
        print("Score after stage 1-2 event : %.8f" % score)

        return self.clusters

        # stage 2-1
        labels = np.unique(self.clusters)
        self._eliminate_outliers(hits, labels, X, stage=2)

        one_submission = create_one_event_submission(event_id, hits, self.clusters)
        dataset_submissions.append(one_submission)
        score = score_event(truth, one_submission)
        print("Score after eliminate stage 2-1 event : %.8f" % score)

        max_len = np.max(self.clusters)
        mask = self.clusters == 0
        print("outlier ")
        print(len(mask))
        self.clusters[mask] = self._init(hits[mask], stage=4) + max_len
        # one_submission= self._extend(one_submission,hits)

        one_submission = create_one_event_submission(event_id, hits, self.clusters)
        # one_submission= self._extend(one_submission,hits)
        dataset_submissions.append(one_submission)
        score = score_event(truth, one_submission)
        print("Score after stage 2-1 event : %.8f" % score)

        # 0-3

        # stage 2-2
        labels = np.unique(self.clusters)
        self._eliminate_outliers(hits, labels, X, stage=3)

        one_submission = create_one_event_submission(event_id, hits, self.clusters)
        dataset_submissions.append(one_submission)
        score = score_event(truth, one_submission)
        print("Score after eliminate stage 2-2 event : %.8f" % score)

        max_len = np.max(self.clusters)
        mask = self.clusters == 0
        print("outlier ")
        print(len(mask))
        self.clusters[mask] = self._init(hits[mask], stage=2) + max_len
        one_submission = create_one_event_submission(event_id, hits, self.clusters)
        one_submission = self._extend(one_submission, hits)
        dataset_submissions.append(one_submission)
        score = score_event(truth, one_submission)
        print("Score after stage 2-1 event : %.8f" % score)

        # stage 3-1
        labels = np.unique(self.clusters)
        self._eliminate_outliers(hits, labels, X, stage=4)

        one_submission = create_one_event_submission(event_id, hits, self.clusters)
        dataset_submissions.append(one_submission)
        score = score_event(truth, one_submission)
        print("Score after eliminate stage3 event : %.8f" % score)

        max_len = np.max(self.clusters)
        mask = self.clusters == 0
        print("outlier ")
        print(len(mask))
        self.clusters[mask] = self._init(hits[mask], stage=3) + max_len
        # self.clusters[mask] = self._init2(hits[mask])+max_len

        return self.clusters

    def predict_merge(self, hits):
        hits = hits.assign(rrr=np.sqrt(hits.x ** 2 + hits.y ** 2))
        return self._init_merge(hits, stage=0)  # 90
