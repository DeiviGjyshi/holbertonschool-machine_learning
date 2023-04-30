#!/usr/bin/env python3
"""Normalization task"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Update momentum"""
    vt = np.multiply(beta1, v) + np.multiply((1- beta1), grad)
    var = var - np.multiply(alpha, vt)
    return (var, vt)
