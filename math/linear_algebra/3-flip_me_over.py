#!/usr/bin/env python3
"""Flip me over"""


def matrix_transpose(matrix):
    """Flip me over"""
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    
    transposed = [[0 for _ in range(num_rows)] for _ in range(num_cols)]
    for i in range(num_rows):
        for j in range(num_cols):
            transposed[j][i] = matrix[i][j]
    
    return transposed
