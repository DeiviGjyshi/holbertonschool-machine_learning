#!/usr/bin/env python3
"""Clusterin tasks"""
import numpy as np


def variance(X, C):
    """Variance function"""
    if type(X) is not np.ndarray or type(C) is not np.ndarray:
        return None
    if len(X.shape) != 2 or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None
    var = np.sum((X - C[:, np.newaxis])**2, axis=-1)
    mean = np.sqrt(var)
    mini = np.min(mean, axis=0)
    var = np.sum(mini ** 2)
    return np.sum(var)
