#!/usr/bin/env python3
import numpy as np
"""Cat got your tongue"""



def np_cat(mat1, mat2, axis=0):
    """Cats got your tongue"""
    arr = np.concatenate((mat1, mat2), axis=axis)
    return arr
