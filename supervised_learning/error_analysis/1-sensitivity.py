#!/usr/bin/env python3
"""Error analysis task"""
import numpy as np


def sensitivity(confusion):
    """Sensitivity"""
    TP = np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    sensitivity = TP / (TP + FN)
    return sensitivity
