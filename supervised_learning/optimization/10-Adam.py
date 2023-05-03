#!/usr/bin/env python3
"""Optimization task"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Adam upgraded"""
    optimizer = tf.train.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon)
    return optimizer.minimize(loss)
