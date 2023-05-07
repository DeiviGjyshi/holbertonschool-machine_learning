#!/usr/bin/env python3
"""Error analysis task"""
import numpy as np


def specificity(confusion):
    """specificity"""
    TP = np.diag(confusion)
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    spec = TN / (TN + FP)
    return spec
