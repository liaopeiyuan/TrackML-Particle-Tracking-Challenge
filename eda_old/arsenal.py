"""
arsenal.py

useful tools for file loading, feature engineering, etc.

by Tianyi Miao

Index:
functions:
get_event_name: accept integer event id, return event name for TrackML helper library
classes:
StaticFeatureEngineer
"""

import os

# define data name strings as constants; prevent spelling errors
HITS = "hits"
CELLS = "cells"
PARTICLES = "particles"
TRUTH = "truth"


def get_directories(parent_dir="./",
                    train_dir="train/",
                    test_dir="test/",
                    detectors_dir="detectors.csv",
                    sample_submission_dir="sample_submission.csv"):
    train_dir = parent_dir + train_dir
    test_dir = parent_dir + test_dir
    detectors_dir = parent_dir + detectors_dir
    sample_submission_dir = parent_dir + sample_submission_dir

    # there are 8850 events in the training dataset; some ids from 1000 to 9999 are skipped
    if os.path.isdir(train_dir):
        train_event_id_list = sorted(set(int(x[x.index("0"):x.index("-")]) for x in os.listdir(train_dir)))
    else:
        train_dir = None
        train_event_id_list = []

    if os.path.isdir(test_dir):
        test_event_id_list = sorted(set(int(x[x.index("0"):x.index("-")]) for x in os.listdir(test_dir)))
    else:
        test_dir = None
        test_event_id_list = []

    if not os.path.exists(detectors_dir):
        detectors_dir = None

    if not os.path.exists(sample_submission_dir):
        sample_submission_dir = None

    return train_dir, test_dir, detectors_dir, sample_submission_dir, train_event_id_list, test_event_id_list


def get_event_name(event_id):
    return "event" + str(event_id).zfill(9)


class StaticFeatureEngineer(object):
    """
    General-purpose feature engineering that does not learn from data (henceforth static)
    Does not consider dataframe merging yet (so only applies to hits_df)
    """
    def __init__(self):
        self.variables = []
        self.methods = []
        self.compiled = False

    def get_n_variables(self):
        if not self.compiled:
            raise RuntimeWarning("StaticFeatureEngineer is not yet compiled. The variables may not be final.")
        return len(self.variables)

    def get_variables(self):
        if not self.compiled:
            raise RuntimeWarning("StaticFeatureEngineer is not yet compiled. The variables may not be final.")
        return self.variables

    def add_method(self, variable, method):
        self.variables.append(variable)
        self.methods.append(method)

    def compile(self):
        self.compiled = True

    def transform(self, df, copy=True):
        if not self.compiled:
            raise RuntimeWarning("StaticFeatureEngineer is not yet compiled. The methods may not be final.")
        if copy:
            df = df.copy()
        for variable, method in zip(self.variables, self.methods):
            df[variable] = method(df)
        return df
