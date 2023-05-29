#!/usr/bin/env python3
"""Conovlutional task"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Convolution backward"""
    m, h_new, w_new, c_new = dZ.shape
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
    db = np.sum(dZ, axis=(0,1,2))
    images = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant', constant_values=0)
    dA = np.zeros(images.shape)
    dW = np.zeros(W.shape)
    for k in range(m):
        for i in range(h_new):
            x = i * sh
            for j in range(w_new):
                y = j * sw
                for g in range(c_new):
                    A = images[k, x:x + kh, y:y + kw, :]
                    Wmin = W[:, :, :, g]
                    dZmin = dZ[k, i, j, g]
                    dA[k, x:x + kh, y:y + kw, :] += dZmin * Wmin
                    dW[:, :, :, g] += dZmin * A
    if padding == "same":
        dA = dA[:, ph:-ph, pw:-pw, :]
    return(dA, dW, db)
