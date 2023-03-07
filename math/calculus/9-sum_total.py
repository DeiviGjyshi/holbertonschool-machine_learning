#!/usr/bin/env python3
"""Sum total"""


def summation_i_squared(n):
    """Sum total"""
    if not isinstance(n, int) or n <= 0:
        return None
    return int(n * (n + 1) * (2 * n + 1) / 6)
