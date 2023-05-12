#!/usr/bin/env python3
"""L2 regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """L2 regularization layer"""
    W = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l2 = tf.contrib.layers.l2_regularizer(lambtha)
    model = tf.layers.Dense(units=n, activation=activation,
                            name="layer", kernel_initializer=W,
                            kernel_regularizer=l2)
    return model(prev)
