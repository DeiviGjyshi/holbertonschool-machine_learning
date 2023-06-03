#!/usr/bin/env python3
"""Deep neural architecture"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Identity block"""
    F11, F3, F12 = filters
    c1 = K.layers.Conv2D(filters=F11,
                         kernel_size=(1, 1),
                         padding='same',
                         kernel_initializer='he_normal')(A_prev)
    c1 = K.layers.BatchNormalization(axis=3)(c1)
    c1 = K.layers.Activation('relu')(c1)
    c1 = K.layers.Conv2D(filters=F3,
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer='he_normal')(c1)
    c1 = K.layers.BatchNormalization(axis=3)(c1)
    c1 = K.layers.Activation('relu')(c1)
    c1 = K.layers.Conv2D(filters=F12,
                         kernel_size=(1, 1),
                         padding='same',
                         kernel_initializer='he_normal')(c1)
    c1 = K.layers.BatchNormalization(axis=3)(c1)
    c1 = K.layers.Add()([c1, A_prev])
    c1 = K.layers.Activation('relu')(c1)
    return c1
