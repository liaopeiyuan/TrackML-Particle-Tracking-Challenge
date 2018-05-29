"""
session.py

- class
Session: a driver class for TrackML scripts
many of its functions are copied from the old arsenal.py

"""

import os


class Session(object):
    HITS = "hits"
    CELLS = "cells"
    PARTICLES = "particles"
    TRUTH = "truth"

    def __init__(self, parent_dir=None, train_dir=None, test_dir=None, detectors_dir=None, sample_submission_dir=None):
        self._init_dir(parent_dir, train_dir, test_dir, detectors_dir, sample_submission_dir)

    def _init_dir(self, parent_dir, train_dir, test_dir, detectors_dir, sample_submission_dir):
        """
        Initialize directories for the dataset
        parent_dir: the parent directory that contains train (directory), test (directory), detectors (csv file), and
        sample_submission (csv file)
        The following directories are all relative to parent_dir:
        train_dir, test_dir, detectors_dir, sample_submission_dir
        """
        if isinstance(parent_dir, str):
            self._parent_dir = parent_dir
        else:
            self._parent_dir = "./"

        if isinstance(train_dir, str):
            self._train_dir = train_dir
        else:
            self._train_dir = "train/"

        if isinstance(test_dir, str):
            self._test_dir = test_dir
        else:
            self._test_dir = "test/"

        if isinstance(detectors_dir, str):
            self._detectors_dir = detectors_dir
        else:
            self._detectors_dir = "detectors.csv"

        if isinstance(sample_submission_dir, str):
            self._sample_submission_dir = sample_submission_dir
        else:
            self._sample_submission_dir = "sample_submission.csv"

    def _init_file(self):
        """
        initialize the data files (train events, test events, detectors, sample_submission, etc.)
        """
        



    @property
    def parent_dir(self):
        return self._parent_dir

    @property
    def train_dir(self):
        return self._train_dir

    @property
    def test_dir(self):
        return self._test_dir

    @property
    def detectors_dir(self):
        return self._detectors_dir

    @property
    def sample_submission_dir(self):
        return self._sample_submission_dir


