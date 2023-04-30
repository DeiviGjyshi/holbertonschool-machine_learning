#!/usr/bin/env python3
"""Normalization task"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Update Adam"""
    m = (beta1 * v) + ((1 - beta1) * grad)
    v = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    mt = np.divide(m, (1 - np.power(beta1, t)))
    vt = np.divide(v, (1 - np.power(beta2, t)))
    var = var - np.multiply(alpha, np.divide(mt, ((vt ** 0.5) + epsilon)))
