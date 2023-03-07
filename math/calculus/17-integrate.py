#!/usr/bin/env python3
"""Task 17 integrate"""


def poly_integral(poly, C=0):
    """Task 17 integrate"""
    if not isinstance(C, int) or not isinstance(poly, list) or len(poly) == 0:
        return None
    k = len(poly)
    res = []
    res.append(C)
    for j in range(k):
       res.append(poly[j] / (j + 1) if poly[j] % (j + 1) != 0 else int(poly[j] / (j + 1)))
    return res
