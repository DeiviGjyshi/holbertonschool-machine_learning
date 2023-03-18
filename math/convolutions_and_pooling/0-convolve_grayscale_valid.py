#!/usr/bin/env python3
"""Convolutions task 0"""


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    convoluted = np.zeros((m, h - kh + 1, w - kw + 1))
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            output = np.sum(images[:, i: i + kh, j: j + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, i, j] = output
    return convoluted
