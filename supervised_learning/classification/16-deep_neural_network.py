i#!/usr/bin/env python3
"""Deep neural network 1"""
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network class"""
    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        self.weights["W1"] = np.random.randn(layers[0], nx) * np.sqrt(2 / nx)
        self.weights["b1"] = np.zeros([layers[0], 1], dtype=float)
        if layers[0] <= 0 or type(layers[0]) is not int:
            raise TypeError("layers must be a list of positive integers")
        for i in range(1, self.L):
            if i <= 0:
                raise TypeError("layers must be a list of positive integers")
            self.weights["W{}".format(i + 1)] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.weights["b{}".format(i + 1)] = np.zeros([layers[i], 1], dtype=float)
