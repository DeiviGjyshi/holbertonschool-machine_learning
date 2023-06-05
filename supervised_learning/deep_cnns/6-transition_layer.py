#!/usr/bin/env python3
"""Deep neural architecture"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Transition layer"""
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    size = int(nb_filters * compression)
    X = K.layers.Conv2D(size, kernel_size=(1, 1),
                        padding='same',
                        kernel_initializer='he_normal')(X)
    X = K.layers.AvgPool2D((2, 2), padding='same')(X)
    return (X, size)
