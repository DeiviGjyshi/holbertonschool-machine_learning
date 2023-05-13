#!/usr/bin/env python3
"""Regularization taksk"""
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Early stopping"""
    stop = False
    if opt_cost - cost > threshold:
        count = 0
    else:
        count = count + 1
    if count == patience:
        stop = True
    return (stop, count)
