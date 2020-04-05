# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:41:35 2020

@author: javier.moral.hernan1
"""
import numpy as np
from sklearn.utils import class_weight


def xgboostWeightedData(y):
    sample_weights = list(class_weight
                          .compute_sample_weight('balanced', y))
    return sample_weights


def CreateBalancedSampleWeights(y_train, largest_class_weight_coef):
    classes = np.unique(y_train, axis=0)
    classes.sort()
    class_samples = np.bincount(y_train)
    total_samples = class_samples.sum()
    n_classes = len(class_samples)
    weights = total_samples / (n_classes * class_samples * 1.0)
    class_weight_dict = {key: value
                         for (key, value) in zip(classes, weights)}
    class_weight_dict[classes[1]] = class_weight_dict[
        classes[1]] * largest_class_weight_coef
    sample_weights = [class_weight_dict[y] for y in y_train]
    return sample_weights
