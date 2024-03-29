#!/usr/bin/env python3
"""RNNs network"""
import numpy as np


def softmax(x):
    """softmax function"""
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def sigmoid(x):
    """Sigmoid"""
    return (1 / (1 + np.exp(-x)))


class LSTMCell:
    """LSTM cell"""
    def __init__(self, i, h, o):
        """Constructor"""
        self.Wf = np.random.normal(size=(i+h, h))
        self.Wu = np.random.normal(size=(i+h, h))
        self.Wc = np.random.normal(size=(i+h, h))
        self.Wo = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Forward prop RNN"""
        concat = np.hstack((h_prev, x_t))
        f = sigmoid(np.dot(concat, self.Wf) + self.bf)
        u = sigmoid(np.dot(concat, self.Wu) + self.bu)
        c_bar = np.tanh(np.dot(concat, self.Wc) + self.bc)
        c_next = f * c_prev + u * c_bar
        o = sigmoid(np.dot(concat, self.Wo) + self.bo)
        h_next = o * np.tanh(c_next)
        y = softmax(np.dot(h_next, self.Wy) + self.by)
        return (h_next, c_next, y)
