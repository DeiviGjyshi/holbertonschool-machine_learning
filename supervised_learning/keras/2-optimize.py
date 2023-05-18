#!/usr/bin/env python3
"""Keras task"""
from tensorflow import keras


def optimize_model(network, alpha, beta1, beta2):
    """Model optimizer"""
    optim = keras.optimizers.Adam(learning_rate=alpha,
                                  beta_1=beta1, beta_2=beta2)
    network.compile(loss="categorical_crossentropy",
                    optimizer=optim, metrics=["accuracy"])
    return None
