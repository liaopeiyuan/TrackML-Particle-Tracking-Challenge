from session import *

import gc
import numpy as np
import pandas as pd
from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

import os
import timeit
from multiprocessing import Pool


from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from trackml.dataset import load_event
from trackml.score import score_event



np.random.seed()  # restart random number generator
s1 = Session(parent_dir="/rscratch/xuanyu/KAIL/test_trackml/")
n_events = 125
test_dataset_submissions=[]

#list_of_test_events = s1.get_test_event(n=n_events, content=[s1.HITS], randomness=True)[1]
i=0
for event_id, hits, cells in tqdm(load_dataset("/rscratch/xuanyu/KAIL/test_trackml/test", parts=['hits', 'cells'])):
# Track pattern recognition
    #print(hits)
    print('Event ID: ', event_id)
    labels = s1.predict(hits)

    # Prepare submission for an event
    one_submission = create_one_event_submission(event_id, hits, labels)

    for i in range(4): 
        one_submission = s1._extend(one_submission, hits)
    test_dataset_submissions.append(one_submission)

    if i==3 or i%20==0:
        submission = pd.concat(test_dataset_submissions, axis=0)
        submission.to_csv('submission_600_temp.csv', index=False)
    print('Event ID: ', event_id)
    #del model
    del labels
    gc.collect()

# Create submission file
submission = pd.concat(test_dataset_submissions, axis=0)
submission.to_csv('submission_600.csv', index=False)
