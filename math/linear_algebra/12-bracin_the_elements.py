#!/usr/bin/env python3
"""Bracin elements"""


def np_elementwise(mat1, mat2):
    """Bracin elements"""
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return (add, sub, mul, div)
