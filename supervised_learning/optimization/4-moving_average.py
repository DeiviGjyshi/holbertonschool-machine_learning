#!/usr/bin/env python3
"""Weighted moving average"""
import matplotlib.pyplot as plt
import numpy as np


def moving_average(data, beta):
    """Moving average"""
    T = [0]
    for i in range(len(data)):
        T.append((beta * T[i]) + ((1 - beta) * data[i]))
    m = []
    for i in range(1, len(T)):
        m.append(T[i] / (1 - (beta ** i)))
    return m
