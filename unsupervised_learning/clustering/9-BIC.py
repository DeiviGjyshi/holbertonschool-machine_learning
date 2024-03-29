#!/usr/bin/env python3
"""Clusterin tasks"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """BIC function"""
    if type(X) is not np.ndarray or type(kmin) is not int:
        return (None, None, None, None)
    if len(X.shape) != 2:
        return (None, None, None, None)
    if type(iterations) is not int or iterations <= 0:
        return (None, None, None, None)
    if type(kmax) is not int or kmax <= 0 or kmax >= X.shape[0]:
        return (None, None, None, None)
    if kmin <= 0 or kmin >= X.shape[0] or kmin >= kmax:
        return (None, None, None, None)
    if type(tol) is not float or tol <= 0:
        return (None, None, None, None)
    if type(verbose) is not bool:
        return (None, None, None, None)
    n, d = X.shape
    ki = []
    li = []
    bi = []
    tup = []
    for k in range(kmin, kmax + 1):
        pi, m, S, g, ll = expectation_maximization(X, k, iterations,
                                                   tol, verbose)
        p = (d * k) + (k * d * (d + 1) / 2) + k - 1
        li.append(ll)
        ki.append(k)
        tup.append((pi, m, S))
        BIC = p * np.log(n) - 2 * ll
        bi.append(BIC)
    ll = np.array(li)
    b = np.array(bi)
    top = np.argmin(b)
    return (ki[top], tup[top], ll, b)
