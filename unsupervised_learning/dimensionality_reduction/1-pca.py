#!/usr/bin/env python3
"""Dimensionality reduction"""
import numpy as np


def pca(X, ndim):
    """Function PCA v2"""
    data = X - np.mean(X, axis=0)
    u, S, v = np.linalg.svd(data)
    W = v[:ndim].T
    return np.matmul(data, W)
