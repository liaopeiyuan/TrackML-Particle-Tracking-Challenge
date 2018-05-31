"""
session.py

- class
Session: a driver class for TrackML scripts
many of its functions are copied from the old arsenal.py

"""

import os

import numpy as np
import pandas as pd

from trackml.dataset import load_event


class Session(object):
    HITS = "hits"
    CELLS = "cells"
    PARTICLES = "particles"
    TRUTH = "truth"

    def __init__(self, parent_dir="./", train_dir="train/", test_dir="test/", detectors_dir="detectors.csv", sample_submission_dir="sample_submission.csv"):
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
        return event_names,\
            (load_event(self._parent_dir + self._train_dir + event_name, content) for event_name in event_names)

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


if __name__ == "__main__":
    s1 = Session(parent_dir="E:/TrackMLData/")
    event_names, event_loaders = s1.remove_train_events(4, content=[s1.HITS, s1.TRUTH], randomness=True)
    for hits, truth in event_loaders:
        print(hits)
        print("=" * 120)
        print(truth)
