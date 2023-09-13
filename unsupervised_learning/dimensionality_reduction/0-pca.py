#!/usr/bin/env python3
"""Dimensionality reduction"""
import numpy as np


def pca(X, var=0.95):
    """PCA"""
    U, S, V = np.linalg.svd(X)
    acum = np.cumsum(S)
    dim = []
    for i in range(len(S)):
        if ((acum[i]) / acum[-1]) >= var:
            dim.append(i)
    r = dim[0] + 1
    return V.T[:, :r]
