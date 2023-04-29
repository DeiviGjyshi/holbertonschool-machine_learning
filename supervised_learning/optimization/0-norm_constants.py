#!/usr/bin/env python3
"""Normalization task"""
import numpy as np


def normalization_constants(X):
    """Normalization constants"""
    m, nx = X.shape
    mean = sum(X) / m
    X = X - mean
    st_dev = (sum(X ** 2) / m) ** 0.5
    return (mean, st_dev)
