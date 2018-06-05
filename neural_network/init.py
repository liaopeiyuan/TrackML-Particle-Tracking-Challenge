# Alexander's Linux environment
DATA_DIR ='/home/alexanderliao/data/Kaggle/competitions/trackml-particle-identification'
TEST_DIR = '/home/alexanderliao/data/GitHub/Kaggle-TrackML/portable-dataset'
RESULTS_DIR = '/home/alexanderliao/data/GitHub/Kaggle-TrackML/neural_network/results'

##---------------------------------------------------------------------
import os
from datetime import datetime
PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))
IDENTIFIER   = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

#numerical libs
"""
 Everything you need for neural network

 Author: Peiyuan (Alexander) Liao
"""
import math
import numpy as np
import random
import PIL
import cv2

import matplotlib
matplotlib.use('TkAgg')
#matplotlib.use('WXAgg')
#matplotlib.use('Qt4Agg')
#matplotlib.use('Qt5Agg') #Qt4Agg

# Pytorch Libraries
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel


# Standard  Libraries
import collections
import copy
import numbers
import inspect
import shutil
from timeit import default_timer as timer

import csv
import pandas as pd
import pickle
import glob
import sys
from distutils.dir_util import copy_tree
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sklearn.preprocessing
from sklearn.cluster import dbscan
from sklearn.preprocessing import scale

from trackml.score import score_event



# KAIL libraries
from utils.session import Session
from geometric.helix import HelixUnroll
from geometric.tools import merge_naive, reassign_noise, label_encode, hit_completeness



from geometric.tools import merge_naive, merge_discreet
from geometric.helix import HelixUnroll
from utils.session import Session





# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def np_sigmoid(x):
  return 1 / (1 + np.exp(-x))


def load_pickle_file(pickle_file):
    with open(pickle_file,'rb') as f:
        x = pickle.load(f)
    return x

def save_pickle_file(pickle_file, x):
    with open(pickle_file, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)

def fast_score(df, pred):
    return score_event(
        truth=df,
        submission=pd.DataFrame({"hit_id": df.hit_id, "track_id": pred})
    )


def get_psi_slice(df, lo, hi):
    df.loc[:, "psi"] = np.arctan2(np.sqrt(df.x ** 2 + df.y ** 2), np.abs(df.z))
    idx = (df.psi >= np.deg2rad(lo)) & (df.psi < np.deg2rad(hi))
    best_cluster = label_encode(reassign_noise(df.particle_id, ~idx))
    best_score = fast_score(df, best_cluster)  # the best possible score achievable by the helix unrolling algorithm
    print("psi=[{}, {}), best possible score={:.6f}".format(lo, hi, best_score))


if __name__ == "__main__":
    print ('matplotlib backend is : {}'.format(matplotlib.get_backend()))

    SEED = int(time.time()) 
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print ('\tset random seed')
    print ('\t\tSEED=%d'%SEED)


    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled   = True
    print ('\tset cuda environment')
    print ('\t\ttorch.__version__              =', torch.__version__)
    print ('\t\ttorch.version.cuda             =', torch.version.cuda)
    print ('\t\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version())

    print ('\t\ttorch.cuda.device_count()      =', torch.cuda.device_count())
    print ('\t\ttorch.cuda.current_device()    =', torch.cuda.current_device())
