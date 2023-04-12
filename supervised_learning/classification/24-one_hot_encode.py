#!/usr/bin/env python3
"""One hot encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """One hot encode"""
    m = Y.shape[0]
    one_hot_y = np.zeros((classes, m))
    one_hot_y[np.arange(classes), Y] = 1
    return one_hot_y
