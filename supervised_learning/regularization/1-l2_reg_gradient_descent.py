#!/usr/bin/env python3
"""Regularization taksk"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """L2 Gradient descent"""
    A2 = cache["A{}".format(L)]
    dz = A2 - Y
    for i in range(L, 0, -1):
        L2_re = (lambtha / Y.shape[1]) * weights["W{}".format(i)]
        db = (np.sum(dz, axis=1, keepdims=True) / Y.shape[1])
        dw = (np.matmul(dz, cache["A{}".format(i - 1)].T) / Y.shape[1]) + L2_re
        dz = np.matmul(weights["W" + str(
            i)].T, dz) * (1 - (cache["A" + str(
                i - 1)] ** 2))
        weights["b{}".format(i)] = weights["b{}".format(
            i)] - (alpha * db)
        weights["W{}".format(i)] = weights["W{}".format(
            i)] - (alpha * dw)
