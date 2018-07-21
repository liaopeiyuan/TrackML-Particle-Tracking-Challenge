"""
session.py
by Tianyi Miao

- class
Session

- main

"""

import os

import numpy as np
import pandas as pd

from trackml.dataset import load_event

class Session(object):
    """
    A highly integrated framework for efficient data loading, prediction submission, etc. in TrackML Challenge
    (improved version of the official TrackML package)

    Precondition: the parent directory must be organized as follows:
    - train (directory)
        - event000001000-cells.csv
        ...
    - test (directory)
        - event000000001-cells.csv
        ...
    - detectors.csv
    - sample_submission.csv
    """
    # important constants to avoid spelling errors
    HITS = "hits"
    CELLS = "cells"
    PARTICLES = "particles"
    TRUTH = "truth"

    def __init__(self, parent_dir="./", train_dir="train/", test_dir="test/", detectors_dir="detectors.csv",
                 sample_submission_dir="sample_submission.csv"):
        """
        default input:
        Session("./", "train/", "test/", "detectors.csv", "sample_submission.csv")
        Session(parent_dir="./", train_dir="train/", test_dir="test/", detectors_dir="detectors.csv", sample_submission_dir="sample_submission.csv")
        """
        self._parent_dir = parent_dir
        self._train_dir = train_dir
        self._test_dir = test_dir
        self._detectors_dir = detectors_dir
        self._sample_submission_dir = sample_submission_dir

        if not os.path.isdir(self._parent_dir):
            raise ValueError("The input parent directory {} is invalid.".format(self._parent_dir))

        # there are 8850 events in the training dataset; some ids from 1000 to 9999 are skipped
        if os.path.isdir(self._parent_dir + self._train_dir):
            self._train_event_id_list = sorted(
                set(int(x[x.index("0"):x.index("-")]) for x in os.listdir(self._parent_dir + self._train_dir)))
        else:
            self._train_dir = None
            self._train_event_id_list = []

        if os.path.isdir(self._parent_dir + self._test_dir):
            self._test_event_id_list = sorted(
                set(int(x[x.index("0"):x.index("-")]) for x in os.listdir(self._parent_dir + self._test_dir)))
        else:
            self._test_dir = None
            self._test_event_id_list = []

        if not os.path.exists(self._parent_dir + self._detectors_dir):
            self._detectors_dir = None

        if not os.path.exists(self._parent_dir + self._sample_submission_dir):
            self._sample_submission_dir = None

    @staticmethod
    def get_event_name(event_id):
        return "event" + str(event_id).zfill(9)

    def get_train_events(self, n=10, content=(HITS, TRUTH), randomness=True):
        n = min(n, len(self._train_event_id_list))
        if randomness:
            event_ids = np.random.choice(self._train_event_id_list, size=n, replace=False).tolist()
        else:
            event_ids, = self._train_event_id_list[:n]
            self._train_event_id_list = self._train_event_id_list[n:] + self._train_event_id_list[:n]

        event_names = [Session.get_event_name(event_id) for event_id in event_ids]
        return event_names, \
            (load_event(self._parent_dir + self._train_dir + event_name, content) for event_name in event_names)

    def remove_train_events(self, n=10, content=(HITS, TRUTH), randomness=True):
        """
        get n events from self._train_event_id_list:
        if random, get n random events; otherwise, get the first n events
        :return:
         1. ids: event ids
         2. an iterator that loads a tuple of hits/cells/particles/truth files
        remove these train events from the current id list
        """
        n = min(n, len(self._train_event_id_list))
        if randomness:
            event_ids = np.random.choice(self._train_event_id_list, size=n, replace=False).tolist()
            for event_id in event_ids:
                self._train_event_id_list.remove(event_id)
        else:
            event_ids, self._train_event_id_list = self._train_event_id_list[:n], self._train_event_id_list[n:]

        event_names = [Session.get_event_name(event_id) for event_id in event_ids]
        return event_names, \
            (load_event(self._parent_dir + self._train_dir + event_name, content) for event_name in event_names)

    def get_test_event(self, n=3, content=(HITS, TRUTH), randomness=True):
        n = min(n, len(self._test_event_id_list))
        if randomness:
            event_ids = np.random.choice(self._test_event_id_list, size=n, replace=False).tolist()
        else:
            event_ids, = self._test_event_id_list[:n]
            self._test_event_id_list = self._test_event_id_list[n:] + self._test_event_id_list[:n]

        event_names = [Session.get_event_name(event_id) for event_id in event_ids]
        return event_names, \
            (load_event(self._parent_dir + self._test_dir + event_name, content) for event_name in event_names)
    
    def remove_test_events(self, n=10, content=(HITS, CELLS), randomness=False):
        n = min(n, len(self._test_event_id_list))
        if randomness:
            event_ids = np.random.choice(self._test_event_id_list, size=n, replace=False).tolist()
            for event_id in event_ids:
                self._test_event_id_list.remove(event_id)
        else:
            event_ids, self._test_event_id_list = self._test_event_id_list[:n], self._test_event_id_list[n:]
        event_names = [Session.get_event_name(event_id) for event_id in event_ids]
        return event_names, \
            (load_event(self._parent_dir + self._test_dir + event_name, content) for event_name in event_names)

    def make_submission(self, predictor, path):
        """
        :param predictor: function, predictor(hits: pd.DataFrame, cells: pd.DataFrame)->np.array
         takes in hits and cells data frames, return a numpy 1d array of cluster ids
        :param path: file path for submission file
        """
        sub_list = []  # list of predictions by event
        for event_id in self._test_event_id_list:
            event_name = Session.get_event_name(event_id)

            hits, cells = load_event(self._parent_dir + self._test_dir + event_name, (Session.HITS, Session.CELLS))
            pred = predictor(hits, cells)  # predicted cluster labels
            sub = pd.DataFrame({"hit_id": hits.hit_id, "track_id": pred})
            sub.insert(0, "event_id", event_id)
            sub_list.append(sub)
        final_submission = pd.concat(sub_list)
        final_submission.to_csv(path, sep=",", header=True, index=False)


if __name__ == "__main__":
    s1 = Session(parent_dir="/mydisk/TrackML-Data/tripletnet/")
    event_names, event_loaders = s1.remove_train_events(4, content=[s1.HITS, s1.TRUTH], randomness=True)
