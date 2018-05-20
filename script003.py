"""
script003.py

continue with hidden space strategy
important: error analysis by parts

by Tianyi Miao
"""

import os

import numpy as np
import pandas as pd

import itertools

from sklearn import cluster
from sklearn.preprocessing import StandardScaler
import hdbscan

from trackml.dataset import load_event
from trackml.score import score_event

from arsenal import get_directories, get_event_name, StaticFeatureEngineer
from arsenal import HITS, CELLS, PARTICLES, TRUTH


# define important directories; change it if you store your data differently!
# type help(get_directories) for more information
# TRAIN_DIR, TEST_DIR, DETECTORS_DIR, SAMPLE_SUBMISSION_DIR, TRAIN_EVENT_ID_LIST, TEST_EVENT_ID_LIST = get_directories("E:/TrackMLData/")
TRAIN_DIR, TEST_DIR, DETECTORS_DIR, SAMPLE_SUBMISSION_DIR, TRAIN_EVENT_ID_LIST, TEST_EVENT_ID_LIST = get_directories()


