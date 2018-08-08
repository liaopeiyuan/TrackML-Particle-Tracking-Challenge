"""
an end-to-end encapsulation of helix rotation and clustering.
Use generator semantics, convenient for parallel computation

"""

import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from functools import reduce
from tqdm import tqdm


def dfh_gen_1(df, coef, n_steps=225, mm=1, stepii=4e-6, z_step=0.5):
    """
    default code provided by Alex on Slack, August 2nd
    """
    for z0 in np.arange(-5.5, 5.5, z_step):
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
                df['a1'] = df['a0'].values - np.nan_to_num(np.arccos(mm * ii * df['rt'].values))
                df['sina1'] = np.sin(df['a1'])
                df['cosa1'] = np.cos(df['a1'])
                ss = StandardScaler()
                dfs = ss.fit_transform(df[['sina1', 'cosa1', 'zdivrt', 'zdivr', 'xdivr', 'ydivr']].values)
                # dfs = scale_ignore_nan(dfh[['sina1','cosa1','zdivrt','zdivr','xdivr','ydivr']])
                dfs = np.multiply(dfs, coef)
                yield dfs


def clusterer_gen_1(db_step=5, n_steps=225, adaptive_eps_coef=1, eps=0.05, min_samples=1, metric="euclidean", p=2, n_jobs=1):
    """
    default code provided by Alex on Slack, August 2nd
    """
    for db in np.arange(min_samples, 10, db_step):
        for ii in range(1, n_steps + 1):
            for jj in range(2):
                eps_new = eps + ii * adaptive_eps_coef * 1e-5
                yield DBSCAN(eps=eps_new, min_samples=db, n_jobs=n_jobs, metric=metric, metric_params=None, p=p)


def pred_wrapper(arg):
    return arg[1].fit_predict(arg[0])


def run_helix_cluster(dfh_gen, clusterer_gen, parallel=True):
    if parallel:
        return list(mp.Pool().map(pred_wrapper, zip(dfh_gen, clusterer_gen)))
    else:
        return list(map(pred_wrapper, zip(dfh_gen, clusterer_gen)))


if __name__ == '__main__':
    hits = pd.DataFrame()
    c = [1.5, 1.5, 0.73, 0.17, 0.027, 0.027]
    dfh_gen_1(hits, coef=c, n_steps=225, mm=1, stepii=4e-6, z_step=0.5)
    clusterer_gen_1(db_step=5, n_steps=225, adaptive_eps_coef=1, eps=0.0048, min_samples=1, metric="euclidean", p=2, n_jobs=1)
    
