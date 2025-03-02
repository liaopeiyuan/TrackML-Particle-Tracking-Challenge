# Importing necessary python native package
import gc
import os

import numpy as np
import pandas as pd
from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

# Import self defined packages
from finalAtom.utils.clusterer import Clusterer
from finalAtom.utils.create_submission import create_one_event_submission

#TODO change the submissoincsv save path [include / at the end]
csv_save_path = "/rscratch/xuanyu/aaronmao/"

#TODO change the base path to your sets and train [include / at the end]
base_path = "/rscratch/xuanyu/aaronmao/"
path_to_train = base_path+"train/"
path_to_test = base_path+"test/"

event_prefix = "event000001000"
event_id = "000001000"




def cossimilar(X, Y):
    lx = np.sqrt(np.inner(X, X))
    ly = np.sqrt(np.inner(Y, Y))
    np.inner(X, Y)
    cossimilar = np.inner(X, Y) / (lx * ly)
    return cossimilar


if __name__ == "__main__":

    dataset_submissions = []
    dataset_scores = []

    # for event_id, hits, cells, particles, truth in load_dataset(path_to_train, skip=0, nevents=5):
    hits, cells, particles, truth = load_event(os.path.join(path_to_train, event_prefix))

    # Track pattern recognition
    model = Clusterer(event_id, event_prefix, path_to_train)
    print("after cluster")
    labels = model.predict(hits)
    print("after predict")

    # Prepare submission for an event
    one_submission = create_one_event_submission(0, hits, labels)
    dataset_submissions.append(one_submission)
    # Score the event
    score = score_event(truth, one_submission)
    dataset_scores.append(score)
    # Print out the score for eval
    print("Score for predict event : %.8f" % (score))

    # Track extension
    for i in range(8):
        one_submission = model._extend(one_submission, hits)
        dataset_submissions.append(one_submission)
        # Score for the event
        score = score_event(truth, one_submission)
        dataset_scores.append(score)
        print("Score for final extended event :%d %.8f" % (i, score))

    # Delete model/labels and call for garbage collect
    del model
    del labels
    gc.collect()

    # Preparing the test for submission
    test_dataset_submissions = []

    create_submission = True
    if create_submission:
        for event_id, hits, cells in load_dataset(path_to_test, parts=['hits', 'cells']):

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
        submission.to_csv(csv_save_path+'submission_600.csv', index=False)
