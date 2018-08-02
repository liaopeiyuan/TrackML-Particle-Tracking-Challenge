"""
utils.py

useful tools for clustering
"""
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


def label_encode(y):
    return LabelEncoder().fit_transform(y)


def reassign_noise(labels: np.ndarray, mask):
    """
    assign noisy points (labeled with key_value such as -1 or 0) to their own clusters of size 1
    """
    ret = labels.copy()
    ret[mask] = np.arange(np.sum(mask)) + np.max(ret) + 1
    return ret

"""
def merge_naive(pred_1, pred_2, cutoff=20):
    # naive cluster merging:
    # iterate over hits; if a hit belongs to a larger cluster in pred_2, it is reassigned
    if pred_1 is None:
        return pred_2
    c1, c2 = Counter(pred_1), Counter(pred_2)  # track id -> track size
    n1, n2 = np.vectorize(c1.__getitem__)(pred_1), np.vectorize(
        c2.__getitem__)(pred_2)  # hit id -> track size
    pred = pred_1.copy()
    idx = (n2 > n1) & (n2 < cutoff)
    pred[idx] = max(pred_1) + 1 + pred_2[idx]
    return label_encode(pred)
"""


def merge_naive(pred_1, pred_2, cutoff=20):
    if pred_1 is None:
        return pred_2
    d = pd.DataFrame(data={'s1': pred_1, 's2': pred_2})
    d['N1'] = d.groupby('s1')['s1'].transform('count')
    d['N2'] = d.groupby('s2')['s2'].transform('count')
    max_s1 = d['s1'].max() + 1
    cond = np.where((d['N2'].values > d['N1'].values) & (d['N2'].values < cutoff))
    s1 = d['s1'].values
    s1[cond] = d['s2'].values[cond] + max_s1
    return label_encode(s1)


def merge_discreet(pred_1, pred_2, cutoff=21):
    """
    discreet cluster merging (less likely to reassign points)
    iterate over clusters in pred_2; np.sum(n1[idx]) < c2[track]**2 -> pred[idx] = d + track
    this is self-documenting
    """
    if pred_1 is None:
        return pred_2
    c1, c2 = Counter(pred_1), Counter(pred_2)  # track id -> track size
    n1, n2 = np.vectorize(c1.__getitem__)(pred_1), np.vectorize(
        c2.__getitem__)(pred_2)  # hit id -> track size
    pred = reassign_noise(pred_1, n1 > cutoff)
    pred_2 = reassign_noise(pred_2, n2 > cutoff)
    n1[n1 > cutoff] = 1
    n2[n2 > cutoff] = 1
    d = max(pred) + 1
    for track in c2:
        if c2[track] < 3:
            continue
        idx = pred_2 == track
        if np.sum(n1[idx]) < c2[track]**2:
            pred[idx] = d + track
    return label_encode(pred)


def hit_completeness(df, idx, track_size=None):
    """
    (the number of non-noisy hits in the idx) / (the total number of hits from all particles
    that have at least 1 hit in the idx)
    """
    if track_size is None:
        track_size = df.groupby("particle_id")["x"].agg("count")
    num = (df.loc[idx, "particle_id"] != 0).sum()
    all_particles = df.loc[idx, "particle_id"].unique().tolist()
    if 0 in all_particles:
        all_particles.remove(0)
    denom = track_size[all_particles].sum()
    return num / denom


def track_completeness(df, idx):
    """
    (number of tracks with all hits in the region) / (number of tracks that have at least 1 hit in the region)
    idx is a boolean mask over the region
    """
    all_particles = df.loc[idx, "particle_id"].unique().tolist()
    if 0 in all_particles:
        all_particles.remove(0)

    agg_1 = df.loc[idx, :].groupby("particle_id", sort=True)["x"].agg("count")
    if 0 in agg_1:
        agg_1.drop(0, inplace=True)
    agg_2 = df.loc[df.particle_id.isin(all_particles), :].groupby(
        "particle_id", sort=True)["x"].agg("count")
    return np.mean(agg_1 == agg_2)

"""
def helix_error_gn(x, y, z, x0, y0, damp, iter, verbose=False):

    a = np.random.rand()
    b = np.random.rand()
    print(a)
    print(b)

    for i in range(iter):
        x_est = a * np.cos(b * z)+x0
        y_est = a * np.sin(b * z)+y0

        errx = (x.flatten() - x_est)
        erry = (y.flatten() - y_est)

        if np.mean(errx) < 1e-3 and np.mean(erry) < 1e-3:
            break

        Jxa = np.cos(b * z)

        # print(np.dot(z,np.sin(b*z)))
        Jxb = -1*a * z * np.sin(b * z)

        # print(Jxa.shape)
        # print(Jxb.shape)

        Jx = np.vstack((Jxa, Jxb))
        # print(Jx)
        # print(np.dot(np.linalg.inv(np.dot(Jx,Jx.T)),Jx).T)
        gradx = np.dot(np.dot(np.linalg.inv(np.dot(Jx, Jx.T)), Jx), errx)
        # print(gradx)

        a = a - damp*gradx[0]
        b = b - damp*gradx[1]

        if verbose:
            print('a update wrt x on iter {}: {}'.format(i, a))
            print('b update wrt x on iter {}: {}'.format(i, b))

        Jya = np.sin(b * z)
        Jyb = a*z*np.cos(b*z)
        Jy = np.vstack((Jya, Jyb))
        grady = np.dot(np.dot(np.linalg.inv(np.dot(Jy, Jy.T)), Jy), erry)

        a = a - damp*grady[0]
        b = b - damp*grady[1]

        if verbose:
            print('a update wrt y on iter {}: {}'.format(i, a))
            print('b update wrt y on iter {}: {}'.format(i, b))

        
        # J=np.zeros((2,2))
        # J[0,0]=np.sin(b*z)
        # J[0,1]=
        # J[1,0]=
        # J[1,1]=
        

    x_est = a * np.cos(b * z)+x0
    y_est = a * np.sin(b * z)+y0

    errx = (x.flatten() - x_est)
    erry = (y.flatten() - y_est)

    return a, b, errx, erry
"""

if __name__ == "__main__":
    x = np.array([0, 0.5403, -0.4161, -0.9900, -0.6536, 0.2837, 0.9602])
    y = np.array([0, 0.8415, 0.9093, 0.1411, -0.7568, -0.9589, -0.2794])
    z = np.array([0, 1, 2, 3, 4, 5, 6])
    # [a, b, errx, erry] = helix_error_gn(x, y, z, 0, 0, 0.01, 60, verbose=False)
    # print(a)
    # print(b)
    # print(np.mean(errx))
    # print(np.mean(erry))
