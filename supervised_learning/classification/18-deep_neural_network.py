#!/usr/bin/env python3
"""Deep neural network"""
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
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__weights["W1"] = np.random.randn(layers[0], nx) * np.sqrt(2 / nx)
        self.__weights["b1"] = np.zeros([layers[0], 1], dtype=float)
        if layers[0] <= 0 or type(layers[0]) is not int:
            raise TypeError("layers must be a list of positive integers")
        for i in range(1, self.L):
            if layers[i] <= 0 or type(layers[i]) is not int:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W{}".format(
                i + 1)] = np.random.randn(layers[i],
                                          layers[i-1])*np.sqrt(2/layers[i-1])
            self.__weights["b{}".format(
                i + 1)] = np.zeros([layers[i], 1], dtype=float)

    @property
    def L(self):
        return(self.__L)

    @property
    def cache(self):
        return(self.__cache)

    @property
    def weights(self):
        return(self.__weights)

    def forward_prop(self, X):
        """forward propagation multi layer"""
        self.__cache["A0"] = X
        for i in range(1, self.__L + 1):
            zl = np.matmul(self.__weights[f"W{i}"], 
                           self.__cache[f"A{i - 1}"
                                        ]) + self.__weights[f"b{i}"]
            sig = 1 / (1 + (np.exp(-zl)))
            self.__cache["A{}".format(
                i)] = sig
        return(sig, self.__cache)
