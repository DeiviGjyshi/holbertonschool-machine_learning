#!/usr/bin/env python3
"""Classification task 1"""
import numpy as np


class Neuron:
    """Neuron class documentation"""
    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 0:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return(self.__W)

    @property
    def b(self):
        return(self.__b)

    @property
    def A(self):
        return(self.__A)
