#!/usr/bin/env python3
"""keras task"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """Train keras model"""
    history = network.fit(data, labels, 
                          batch_size=batch_size, 
                          epochs=epochs, verbose=verbose, 
                          shuffle=shuffle)
    return history
