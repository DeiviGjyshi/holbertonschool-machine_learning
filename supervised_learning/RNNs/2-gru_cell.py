#!/usr/bin/env python3
"""RNNs networks"""
import numpy as np


def softmax(x):
    """softmax function"""
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def sigmoid(x):
    """Sigmoid"""
    return (1 / (1 + np.exp(-x)))

class GRUCell:
    """GRU cell"""
    def __init__(self, i, h, o):
        """Constructor"""
        self.Wz = np.random.normal(size=(i+h, h))
        self.Wr = np.random.normal(size=(i+h, h))
        self.Wh = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Forward prop RNN"""
        concat = np.hstack((h_prev, x_t))
        z = sigmoid(np.dot(concat, self.Wz) + self.bz)
        r = sigmoid(np.dot(concat, self.Wr) + self.br)
        concat = np.hstack((h_prev * r, x_t))
        c = np.tanh(np.dot(concat, self.Wh) + self.bh)
        h_next = np.multiply(z, c) + np.multiply((1-z), h_prev)
        y = softmax(np.dot(h_next, self.Wy) + self.by)
        return h_next, y
