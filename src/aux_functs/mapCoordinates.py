# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 18:19:37 2020

@author: javier.moral.hernan1
"""

import numpy as np


def scaleTransformation(i, x_min, x_max, y_min, y_max):
    numerator = (i - x_min)*(y_max - y_min)
    denominator = (x_max - x_min)
    transformation = (numerator/denominator) + y_min
    return transformation


def rangeTranfer(values, desired_range):
    x_min = min(values)
    x_max = max(values)
    y_min = desired_range[0]
    y_max = desired_range[1]
    trans = [scaleTransformation(i, x_min, x_max, y_min, y_max)
             for i in values]
    return trans


def getCenterPoint(data, var1, var2):
    centroid = (sum(data[var1]) / len(data),
                sum(data[var2]) / len(data))
    return centroid
