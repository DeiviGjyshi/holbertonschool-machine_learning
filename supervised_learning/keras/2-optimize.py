#!/usr/bin/env python3
"""Keras task"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Optimize keras model"""
    opt = K.optimizers.Adam(alpha, beta1, beta2)
    network.compile(loss='categorical_crossentropy',
                    optimizer=opt, metrics=['accuracy'])
    return None
