#!/usr/bin/env python3
"""Keras task"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """Train keras model"""
    history = network.fit(data, labels,
                          batch_size=batch_size,
                          epochs=epochs, verbose=verbose,
                          validation_data=validation_data,
                          shuffle=shuffle)
    return history
