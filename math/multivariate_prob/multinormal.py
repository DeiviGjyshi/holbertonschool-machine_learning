#!/usr/bin/env python3
"""Multivariate normal"""
import numpy as np


class MultiNormal:
    """Multinormal probability class"""
    def __init__(self, data):
        """Multinormal init function"""
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        n, d = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean, self.cov = self.mean_cov(data.T)
        self.mean = self.mean.reshape(-1, 1)

    def mean_cov(self, X):
        """Mean and convolution"""
        n, d = X.shape
        if type(X) is not np.ndarray or len(X.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if n < 2:
            raise ValueError("data must contain multiple data points")
        mean = np.mean(X, axis=0)
        cov = np.zeros((d, d))
        for i in range(n):
            cov += np.outer((X[i] - mean), (X[i] - mean))
        cov /= (n - 1)
        return mean.reshape(1, d), cov
