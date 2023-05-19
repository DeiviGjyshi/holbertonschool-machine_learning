#!/usr/bin/env python3
"""Keras task"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Predict keras model"""
    prediction = network.predict(data, verbose=verbose)
    return prediction
