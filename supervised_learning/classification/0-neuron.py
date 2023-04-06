#!/usr/bin/env python3
"""Neuron class documentation"""
import numpy as np


class Neuron:
    """ Task 0 neuron class"""
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 0:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
