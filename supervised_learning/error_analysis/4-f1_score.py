#!/usr/bin/env python3
"""Error analysis task"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """F1 score"""
    v = (sensitivity(confusion) ** -1) + (precision(confusion) ** -1)
    return (2 / v)
