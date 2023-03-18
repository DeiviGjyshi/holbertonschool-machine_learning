#!/usr/bin/env python3
"""Convolution with pooling"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Convolution with pooling"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    out_h = int((h  - kh) / sh + 1)
    out_w = int((w - kw) / sw + 1)
    out = np.zeros((m, out_h, out_w, c))
    for i in range(out_h):
        for j in range(out_w):
            if mode == 'avg':
                window = np.mean(images[:, i*sh:i*sh+kh, j*sw:j*sw+kw,:],
                                 axis=(1, 2))
            else:
                window = np.max(images[:, i*sh:i*sh+kh, j*sw:j*sw+kw,:],
                                axis=(1, 2))
            out[:, i, j, :] = window
    return out
