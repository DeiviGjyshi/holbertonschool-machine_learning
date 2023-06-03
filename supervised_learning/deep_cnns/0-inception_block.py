#!/usr/bin/env python3
"""Deep neural architecture"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """inception block"""
    F1, F3R, F3, F5R, F5, FPP = filters
    c1 = K.layers.Conv2D(filters=F1,
                         kernel_size=(1,1),
                         padding='same',
                         activation=K.activations.relu,
                         kernel_initializer=K.initializers.he_normal())
    output_1 = c1(A_prev)
    c2 = K.layers.Conv2D(filters=F3R,
                         kernel_size=(1,1),
                         padding='same',
                         activation=K.activations.relu,
                         kernel_initializer=K.initializers.he_normal())
    output_2 = c2(A_prev)
    c3 = K.layers.Conv2D(filters=F3,
                         kernel_size=(3,3),
                         padding='same',
                         activation=K.activations.relu,
                         kernel_initializer=K.initializers.he_normal())
    output_3 = c3(output_2)
    c4 = K.layers.Conv2D(filters=F5R,
                         kernel_size=(1,1),
                         padding='same',
                         activation=K.activations.relu,
                         kernel_initializer=K.initializers.he_normal())
    output_4 = c4(A_prev)
    c5 = K.layers.Conv2D(filters=F5,
                         kernel_size=(5,5),
                         padding='same',
                         activation=K.activations.relu,
                         kernel_initializer=K.initializers.he_normal())
    output_5 = c5(output_4)
    max_pool = K.layers.MaxPool2D(pool_size=(3,3),
                                  strides=(1,1),
                                  padding='same')
    output_6 = max_pool(A_prev)
    c6 = K.layers.Conv2D(filters=FPP,
                         kernel_size=(1,1),
                         padding='same',
                         activation=K.activations.relu,
                         kernel_initializer=K.initializers.he_normal())
    output_7 = c6(output_6)
    return (K.layers.concatenate([output_1, output_3, output_5, output_7]))
