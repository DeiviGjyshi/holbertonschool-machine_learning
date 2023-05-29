#!/usr/bin/env python3
"""Convolutional task"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Pooling backprop"""
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros(A_prev.shape)
    for g in range(m):
        for i in range(h_new):
            x = i * sh
            for j in range(w_new):
                y = j * sw
                for k in range(c_new):
                    if mode == "max":
                        A_min = A_prev[g, x:x + kh, y:y + kw, k]
                        slope = (A_min == np.max(A_min))
                        dA_prev[g, x:x + kh,
                                y:y + kw, k] += slope * dA[g, i, j, k]
                    else:
                        dAmin = dA[g, i, j, k]/kh/kw
                        dA_prev[g, x:x + kh, y:y + kw, k] += dAmin
    return dA_prev
