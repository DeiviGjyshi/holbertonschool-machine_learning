#!/usr/bin/env python3
"""Convolution same"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Convolution same"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    convoluted = np.zeros((m, h, w))
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            output = np.sum(images[:, i: i + kh, j: j + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, i, j] = output
    return convoluted
