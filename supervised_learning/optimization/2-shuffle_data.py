#!/usr/bin/env python3
"""Normalization task"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffle array"""
    k = X.shape[0]
    rand = np.random.permutation(k)
    return(X[rand], Y[rand])
