#!/usr/bin/env python3
"""Convolution with padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Convolution with padding"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    output_pad = np.zeros((m, h - kh + (2 * ph) + 1, w - kw + (2 * pw) + 1))
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            output = np.sum(images[:, i: i + kh, j: j + kw] * kernel,
                            axis=1).sum(axis=1)
            output_pad[:, i, j] = output
    return output_pad
