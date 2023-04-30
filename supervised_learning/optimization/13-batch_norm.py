#!/usr/bin/env python3
"""Normalization task"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Batch normalization"""
    mean = Z.mean(axis=0)
    var = Z.var(axis=0)
    z1 = (Z - mean) / ((var + epsilon) ** 0.5)
    Znorm = (gamma * z1) + beta
    return Znorm
