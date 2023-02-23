#!/usr/bin/env python3
"""Size me please"""

def matrix_shape(matrix):
    """Size me please"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
