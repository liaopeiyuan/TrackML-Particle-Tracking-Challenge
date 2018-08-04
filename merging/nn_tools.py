"""
code copied from https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
"""

import numpy as np
import pandas as pd
import keras
from sklearn.metrics import precision_recall_fscore_support


class F1Callback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_target = self.validation_data[1]
        val_weight = self.validation_data[2]  # notice: the validation data is weighted
        _val_precision, _val_recall, _val_fscore, _val_support = precision_recall_fscore_support(y_true=val_target, y_pred=val_predict, sample_weight=val_weight, average="binary")
        self.val_f1s.append(_val_fscore)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(f"val_fscore: {_val_fscore}, val_recall: {_val_recall}, val_precision: {_val_precision}")
        # print “ — val_f1: % f — val_precision: % f — val_recall % f” % (_val_f1, _val_precision, _val_recall)
        return


f1_metric = F1Callback()
