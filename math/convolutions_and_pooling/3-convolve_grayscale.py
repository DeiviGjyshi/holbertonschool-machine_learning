#!/usr/bin/env python3
"""Convolution with slide"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Convolution with slide"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh,sw = stride
    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2)
        pw = int(((w - 1) * sw + kw - w) / 2)
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding
    images_padded = np.pad(images, ((0,0),(ph, ph), (pw, pw)), mode='constant')
    out_h = int((h + 2 * ph - kh) / sh + 1)
    out_w = int((w + 2 * pw - kw) / sw + 1)
    output_4 = np.zeros((m, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            var = images_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw]
            output_4[:, i, j] = np.sum(var * kernel, axis=(1, 2))
    return output_4
