#!/usr/bin/env python3
"""Likelihood function"""
import numpy as np


def likelihood(x, n, P):
    """Likelihood function"""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is greater than or/
                         equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for value in P:
        if value > 1 and value < 0:
            raise ValueError()
    factorial = np.math.factorial
    fact_coeff =factorial(n) / (factorial(n-x) * factorial(x))
    likelihood = fact_coeff * (P**x) * ((1-P)**(n-x))
    return likelihood
