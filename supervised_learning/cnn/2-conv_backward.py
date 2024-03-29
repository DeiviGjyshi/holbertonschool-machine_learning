#!/usr/bin/env python3
"""Conovlutional task"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Convolution backward"""
    if padding == 'same':
        ph = int(((A_prev.shape[1] - 1) * stride[0] +
                  W.shape[0] - A_prev.shape[1]) / 2) + 1
        pw = int(((A_prev.shape[2] - 1) * stride[1] +
                  W.shape[1] - A_prev.shape[2]) / 2) + 1
    else:
        ph = 0
        pw = 0
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    npad = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    A_pad = np.pad(A_prev, pad_width=npad,
                   mode='constant', constant_values=0)
    dA = np.zeros(A_pad.shape)
    dW = np.zeros(W.shape)
    for m in range(dZ.shape[0]):
        for i in range(dZ.shape[1]):
            x = i * stride[0]
            for j in range(dZ.shape[2]):
                y = j * stride[1]
                for k in range(dZ.shape[3]):
                    A = A_pad[m, x:x + W.shape[0],
                              y:y + W.shape[1], :]
                    Wmin = W[:, :, :, k]
                    dZmin = dZ[m, i, j, k]
                    dA[m, x:x + W.shape[0],
                       y:y + W.shape[1], :] += dZmin * Wmin
                    dW[:, :, :, k] += dZmin * A
    if padding == 'same':
        dA = dA[:, ph:-ph, pw:-pw, :]
    return (dA, dW, db)
