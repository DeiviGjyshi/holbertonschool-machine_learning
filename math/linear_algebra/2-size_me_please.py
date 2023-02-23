#!/usr/bin/env python
def matrix_shape(matrix):
    mat = []
    mat.append(len(matrix))
    if type(matrix[0]) == list:
        mat.append(len(matrix[0]))
        if type(matrix[0][0]) == list:
            mat.append(len(matrix[0][0]))
    return mat
