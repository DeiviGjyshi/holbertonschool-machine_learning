#!/usr/bin/env python3
"""Regularization taksk"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """L2 regularization"""
    w = 0
    for i in range(1, L + 1):
        w =w + np.linalg.norm(weights['W' + str(i)])
    L2cost = cost + ((lambtha / (2 * m)) * w)
    return L2cost
