#!/usr/bin/env python3
"""Regularization taksk"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """forward propagation multi layer"""
    cache = {}
    cache["A0"] = X
    for i in range(1, L + 1):
        Z = np.dot(weights["W{}".format(i)],
                    cache["A{}".format(
                    i - 1)]) + weights["b{}".format(i)]
        if i != L:
            tanh =np.sinh(Z) / np.cosh(Z)
            D1 = np.random.rand(tanh.shape[0], tanh.shape[1])
            D1 = (D1 < keep_prob).astype(int)
            tanh = tanh * D1
            tanh = tanh / keep_prob
        else:
            t = np.exp(Z)
            tanh = np.exp(Z) / np.sum(t, axis=0, keepdims=True)
        cache["A{}".format(i)] = tanh
        cache["D{}".format(i)] = D1
    return cache