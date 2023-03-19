#!/usr/bin/env python3
"""Convolution same"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Convolution same"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    if kh % 2 == 0:
        kernel = np.vstack((kernel, np.zeros((1, kw))))
        kh += 1
    if kw % 2 == 0:
        kernel = np.hstack((kernel, np.zeros((kh, 1))))
        kw += 1
    ph = max(int((kh - 1) / 2), 0)
    pw = max(int((kw - 1) / 2), 0)
    padded_im = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    convoluted = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            output = np.sum(padded_im[:, i:i+kh, j:j+kw] * kernel,
                            axis=(1, 2))
            convoluted[:, i, j] = output
    return convoluted
