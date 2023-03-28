#!/usr/bin/env python3
"""Multivariate normal"""
import numpy as np


class MultiNormal:

  def __init__(self, data):
    """Multivariate normal"""
    if type(data) is not np.ndarray or len(data.shape) != 2:
        raise TypeError("data must be a 2D numpy.ndarray")
    n, d = data.shape
    if n < 2:
        raise ValueError("data must contain multiple data points")
    self.mean = np.mean(data, axis=0, keepdims=True).T
    self.cov = np.zeros((d, d))
    for i in range(n):
        x = data[i, :].reshape((-1, 1))
        self.cov += (x - self.mean) @ (x - self.mean).T
    self.cov /= (n - 1)
