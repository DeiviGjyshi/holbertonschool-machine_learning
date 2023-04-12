#!/usr/bin/env python3
"""One hot encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """One hot encode"""
    if type(Y) is not np.ndarray or len(Y) == 0:
        return None
    if type(classes) is not int or classes <= Y.max():
        return None
    m = Y.shape[0]
    one_hot_y = np.zeros((m, classes))
    one_hot_y[np.arange(m), Y] = 1
    return one_hot_y.T
