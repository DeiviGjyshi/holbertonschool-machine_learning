#!/usr/bin/env python3
"""Convolution with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Convolution with channels"""
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
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
    out = np.zeros((m, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            window = imp[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            out[:, i, j] = np.sum(window * kernel, axis=(1, 2, 3))
    return out
