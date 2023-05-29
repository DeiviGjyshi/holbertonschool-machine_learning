#!/usr/bin/env python3
"""Convolution neural network"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Covolution forward"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == "valid":
        ph = 0
        pw = 0
    elif padding == "same":
        ph = (((h_prev - 1) * sh) + kh - h_prev) // 2
        pw = (((w_prev - 1) * sw) + kw - w_prev) // 2
    else:
        return
    images = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant', constant_values=0)
    nh = int(((h_prev + (2 * ph) - kh) / sh) + 1)
    nw = int(((w_prev + (2 * pw) - kw) / sw) + 1)
    convoluted = np.zeros((m, nh, nw, c_new))
    for i in range(nh):
        h = i * sh
        for j in range(nw):
            w = j * sw
            for k in range(c_new):
                output = images[:, h:h + kh, w:w + kw, :]
                kernel = W[:, :, :, k]
                convoluted[:, i, j, k] = np.sum(np.multiply(output, kernel),
                                                axis=(1, 2, 3))
    convoluted = convoluted + b
    convoluted = activation(convoluted)
    return convoluted
