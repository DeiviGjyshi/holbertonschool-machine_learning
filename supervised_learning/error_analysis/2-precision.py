#!/usr/bin/env python3
"""Error analysis task"""
import numpy as np


def precision(confusion):
    """Precision"""
    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - np.diag(confusion)
    prec = TP / (TP + FP)
    return prec
