#!/usr/bin/env python3
"""Size me please"""


def matrix_shape(matrix):
    """Size me please"""
    mat = []
    mat.append(len(matrix))
    if type(matrix[0]) == list:
        mat.append(len(matrix[0]))
        if type(matrix[0][0]) == list:
            mat.append(len(matrix[0][0]))
    return mat
