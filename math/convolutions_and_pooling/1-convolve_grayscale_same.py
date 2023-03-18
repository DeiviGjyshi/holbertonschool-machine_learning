#!/usr/bin/env python3
"""Convolution same"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Convolution same"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = max(int((kh - 1) / 2), 0)
    pw = max(int((kw - 1) / 2), 0)
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    convoluted = np.zeros((m, h, w))
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            output = np.sum(padded_images[:, i: i + kh, j: j + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, i, j] = output
    return convoluted
