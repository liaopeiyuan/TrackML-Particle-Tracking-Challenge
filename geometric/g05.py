"""
final geometric unrolling before the end
no z shifting yet

w_xy_r: w_x_r = w_y_r, the weight used in

"""
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def dfh_gen_3(df, w_xy_ra, w_xy_rz, w_xy_c, a_step=1e-5, z_step=0, step_init=0, n_step=100, log=None):
    if log is None or not log:
        log = {"a_delta": [], "z_delta": []}
    # will yield n_step * 2 steps
    df = df[["x", "y", "z"]]
    df["z0"] = df["z"]  # TODO: z-shifting
    df["a0"] = np.arctan2(df["y"], df["x"])  # default angle
    df["ra"] = np.dot(df[["x", "y", "z"]] ** 2, [w_xy_ra, w_xy_ra, 3 - w_xy_ra * 2])  # weighted radius for angle
    df["rz"] = np.dot(df[["x", "y", "z"]] ** 2, [w_xy_rz, w_xy_rz, 3 - w_xy_rz * 2])  # weighted radius for z
    df["rt"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)
    for ii in range(step_init, step_init+n_step):  # step index
        for mm in (-1, 1):  # direction multiplier
            a_delta = mm * ii * a_step
            z_delta = mm * ii * z_step
            log["a_delta"].append(a_delta)
            log["z_delta"].append(z_delta)
            df["a_delta"] = a_delta * df["ra"]
            df["z_delta"] = z_delta * df["rz"]
            df["a1"] = df["a0"] + df["a_delta"]
            df["z1"] = df["z0"] + np.sign(df["z0"]) * df["z_delta"]  # z coordinates are symmetric by the xy-plane
            # perparing features for clustering
            df["sin_a1"] = np.sin(df["a1"])
            df["cos_a1"] = np.cos(df["a1"])
            df["z1_div_rt"] = df["z1"] / df["rt"]
            dfs = StandardScaler().fit_transform(df[["sin_a1", "cos_a1", "z1_div_rt"]])
            dfs = np.multiply(dfs, [w_xy_c, w_xy_c, 3 - w_xy_c * 2])
            yield dfs
    print(f"dfh generator log:\n{log}")


def cluster_gen_3(step_init=0, n_step=225, eps_init=0.05, eps_step=1e-5, min_samples=1, metric="euclidean", p=2, n_jobs=1, log=None):
    if log is None or not log:
        log = {"eps": []}
    for ii in range(step_init, step_init + n_step):
        eps = eps_init + ii * eps_step
        for mm in (-1, 1):
            log["eps"].append(eps)
            yield DBSCAN(eps=eps, min_samples=min_samples, n_jobs=n_jobs, metric=metric, metric_params=None, p=p)
    print(log)


