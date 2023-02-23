#!/usr/bin/env python3
"""Cats got your tongue"""


import numpy as np
def np_cat(mat1, mat2, axis=0):
    """Cats got your tongue"""
    arr = np.concatenate((mat1, mat2), axis = axis)
    return arr
