#!/usr/bin/env python3
"""Intersection function"""
import numpy as np


def intersection(x, n, P, Pr):
    """Intersection function"""
    c = "x must be an integer that is greater than or equal to 0"
    d = "Pr must be a numpy.ndarray with the same shape as P"
    e = "All values in {P} must be in the range [0, 1]"
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(c)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError(d)
    for idx in range(P.shape):
        if P[idx] > 1 or P[idx] < 0:
            raise ValueError(e)
        if Pr[idx] > 1 or Pr[idx] < 0:
            raise ValueError(e)
    if np.isclose([np.sum(Pr)], [1]) == [False]:
         raise ValueError("Pr must sum to 1")
    factorial = np.math.factorial
    fact_coeff =factorial(n) / (factorial(n-x) * factorial(x))
    likelihood = fact_coeff * (P**x) * ((1-P)**(n-x))
    intersection = likelihood * Pr
    return intersection
