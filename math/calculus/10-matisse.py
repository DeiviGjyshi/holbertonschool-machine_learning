#!/usr/bin/env python3
"""Task 10 matisse"""


def poly_derivative(poly):
    """Task 10 matisse"""
    if not isinstance(poly ,list):
        return None
    n = len(poly)
    res = []
    for j in range(n):
        if j == 0:
            continue
        elif j == 1:
            res.append(poly[j])
        elif j > 1:
            res.append(poly[j] * j)
    return res
