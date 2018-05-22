"""
geometric.py

use geometric transformations to solve the problem

by Tianyi Miao
"""

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from trackml.dataset import load_event
from trackml.score import score_event

from arsenal import get_directories, get_event_name, StaticFeatureEngineer
from arsenal import HITS, CELLS, PARTICLES, TRUTH


def helix_1(x, y, z):
    """
    helix unrolling that works since the beginning
    """
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    return x/d, y/d, z/r


def helix_2(x, y, z, theta=20, v=2000):
    """
    rotate the helix by an angle of 20
    theta: angle of rotation
    v: normalization constant
    """
    r = np.sqrt(x ** 2 + y ** 2)
    phi = (theta / 180) * np.pi * (r / v)
    hx = x * np.cos(-phi) - y * np.sin(-phi)
    hy = x * np.sin(-phi) + y * np.cos(-phi)
    return hx, hy, z
