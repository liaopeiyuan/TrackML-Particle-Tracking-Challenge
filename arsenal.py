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
    return parent_dir + train_dir, parent_dir + test_dir, parent_dir + detectors_dir, parent_dir + sample_submission_dir


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
