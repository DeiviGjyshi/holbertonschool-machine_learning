#!/usr/bin/env python3
"""Convolution with multiple kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Convolution with multiple kernels"""
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2)
        pw = int(((w - 1) * sw + kw - w) / 2)
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding
    imp = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
    out_h = int((h + 2 * ph - kh) / sh + 1)
    out_w = int((w + 2 * pw - kw) / sw + 1)
    out_5 = np.zeros((m, out_h, out_w, nc))
    for k in range(nc):
        for i in range(out_h):
            for j in range(out_w):
                window = imp[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
                out_5[:, i, j, k] = np.sum(window * kernels[..., k],
                                           axis=(1, 2, 3))
    return out_5
