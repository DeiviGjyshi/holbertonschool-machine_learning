#!/usr/bin/env python3
"""Normalization task"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """RMS prop update"""
    Et = np.multiply(beta2, s) + np.multiply((1 - beta2), grad ** 2)
    var = var - np.multiply(alpha, np.divide(grad, ((Et ** 0.5) + epsilon)))
    return (var, Et)
