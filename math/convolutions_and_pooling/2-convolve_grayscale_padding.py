#!/usr/bin/env python3
"""Convolution with padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Convolution with padding"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    padded_im = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    output_pad = np.zeros((m, h + 2*ph - kh + 1, w + 2*pw - kw + 1))
    for i in range(h + 2*ph - kh + 1):
        for j in range(w + 2*pw - kw + 1):
            output = np.sum(padded_im[:, i: i + kh, j: j + kw] * kernel,
                            axis=1).sum(axis=1)
            output_pad[:, i, j] = output
    return output_pad
