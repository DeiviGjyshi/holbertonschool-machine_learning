#!/usr/bin/env python3
"""Convolution same"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Convolution same"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = max(int((kh - 1) / 2), 0)
    pw = max(int((kw - 1) / 2), 0)
    padded_im = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    convoluted = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            if kh % 2 == 0 or kw % 2 == 0:
                output = np.sum(images[:, i:i+kh, j:j+kw] * kernel,
                                axis=(1, 2))
            else:
                output = np.sum(padded_im[:, i:i+kh, j:j+kw] * kernel,
                                axis=(1, 2))
            convoluted[:, i, j] = output
    return convoluted
