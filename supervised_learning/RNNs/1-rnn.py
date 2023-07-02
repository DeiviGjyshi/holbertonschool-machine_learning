#!/usr/bin/env python3
"""RNNs networks"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Simple forward prop"""
    T = X.shape[0]
    H = []
    Y = []
    h = h_0
    H.append(h)
    for t in range(T):
        h, y = rnn_cell.forward(h, X[t])
        H.append(h)
        Y.append(y)
    H = np.array(H)
    Y = np.array(Y)
    return (H, Y)
