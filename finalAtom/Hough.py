import Session

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
s1 = Session(parent_dir="/Users/alexanderliao/Documents/GitHub/Kaggle-TrackML/portable-dataset/")
n_events = 125
test_dataset_submissions=[]

list_of_train_events = s1.get_train_events(n=n_events, content=[s1.HITS, s1.TRUTH], randomness=True)[1]
for hits in tqdm(list_of_train_events):
# Track pattern recognition
    model = Clusterer(event_id, event_prefix, path_to_train)
    labels = model.predict(hits)

    # Prepare submission for an event
    one_submission = create_one_event_submission(event_id, hits, labels)

    for i in range(4): 
        one_submission = model._extend(one_submission, hits)
    test_dataset_submissions.append(one_submission)

    print('Event ID: ', event_id)
    del model
    del labels
    gc.collect()

# Create submission file
submission = pd.concat(test_dataset_submissions, axis=0)
submission.to_csv('submission_600.csv', index=False)