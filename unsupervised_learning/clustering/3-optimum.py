#!/usr/bin/env python3
"""Clusterin tasks"""
import numpy as np


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Optimum K"""
    if type(X) is not np.ndarray:
        return (None, None)
    if type(kmin) is not int:
        return (None, None)
    if kmax is not None and type(kmax) is not int:
        return (None, None)
    if kmax is None:
        kmax = X.shape[0]
    if len(X.shape) != 2 or kmin < 1:
        return (None, None)
    if kmax is not None and kmax <= kmin:
        return (None, None)
    if type(iterations) is not int:
        return (None, None)
    if iterations <= 0:
        return (None, None)
    results = []
    variances = []
    k = kmin
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k)
        var = variance(X, C)
        results.append((C, clss))
        variances.append(var)
    first = variances[0]
    d_vars = []
    for i in range(len(variances)):
        d_vars.append(first - variances[i])
    return (results, d_vars)
