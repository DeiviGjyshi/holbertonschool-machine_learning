#!/usr/bin/env python3
"""Error analysis task"""
import numpy as np


def f1_score(confusion):
    """F1 score"""
    f1 = (2 * precision * sensitivity) / (precision + sensitivity)
    return f1
