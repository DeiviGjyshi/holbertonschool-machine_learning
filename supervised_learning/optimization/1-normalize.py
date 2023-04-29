#!/usr/bin/env python3
"""Normalization task"""
import numpy as np


def normalize(X, m, s):
    norm = (X - m) / s
    return norm
