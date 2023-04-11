#!/usr/bin/env python3
"""Neural network task"""
import numpy as np


class NeuralNetwork:
    """NeuralNetwork that defines a neural network"""
    def __init__(self, nx, nodes):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx <= 0:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes <= 0:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return(self.__W1)

    @property
    def b1(self):
        return(self.__b1)

    @property
    def A1(self):
        return(self.__A1)

    @property
    def W2(self):
        return(self.__W2)

    @property
    def b2(self):
        return(self.__b2)

    @property
    def A2(self):
        return(self.__A2)

    def forward_prop(self, X):
        """forward propagation multi layer"""
        z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + (np.exp(-z1)))
        z2 = np.matmul(self.W2, self.A1) + self.b2
        self.__A2 = 1 / (1 + (np.exp(-z2)))
        return (self.A1, self.A2)

    def cost(self, Y, A):
        """Calculate cost"""
        m = Y.shape[1]
        m__loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = (1/m) * (-(m__loss))
        return (cost)

    def evaluate(self, X, Y):
        """Evaluate netwwork"""
        A = self.forward_prop(X)
        a1, a2 = A
        cost = self.cost(Y, a2)
        prediction = np.where(a2 >= 0.5, 1, 0)
        return (prediction, cost)

    def deriv_func(self, Z):
        """Derivative of activation func"""
        z1 = np.matmul(self.W1, Z) + self.b1
        coeff  = 1 / (1 + (np.exp(-z1)))
        deriv = coeff * (1 - coeff)
        return deriv

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """calculate gradient descent"""
        m = Y.shape[1]
        dz2 = (A2 - Y)
        d__W2 = (1 / m) * (np.matmul(A1, dz2.transpose()).transpose())
        d__b2 = (1 / m) * (np.sum(dz2))
        self.__W2 = self.W2 - (alpha * d__W2)
        self.__b2 = self.b2 - (alpha * d__b2)
        dz1 = np.dot(self.__W2.T, dz2) * self.deriv_func(X) 
        d__W1 = (1 / m) * (np.matmul(X, dz1.transpose()).transpose())
        d__b1 = (1 / m) * (np.sum(dz1))
        self.__W1 = self.W1 - (alpha * d__W1)
        self.__b1 = self.b1 - (alpha * d__b1)
