#!/usr/bin/env python3
"""Convolutional neural network"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Pooling layer"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    nh = int(((h_prev - kh) / sh) + 1)
    nw = int(((w_prev - kw) / sw) + 1)
    pooled = np.zeros((m, nh, nw, c_prev))
    for i in range(nh):
        h = i * sh
        for j in range(nw):
            w = j * sw
            output = A_prev[:, h:h + kh, w:w + kw, :]
            if mode == "max":
                pooled[:, i, j, :] = np.max(output, axis=(1,2))
            else:
                pooled[:, i, j, :] = np.average(output, axis=(1,2))
    return pooled
