#!/usr/bin/env python3
"""Error analysis task"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """ Create Confusion"""
    return(np.dot(labels.T, logits))
