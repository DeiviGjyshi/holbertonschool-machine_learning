#!/usr/bin/env python3
"""Optimization task"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """RMS prop upgraded"""
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2,
                epsilon=epsilon)
    train_op = optimizer.minimize(loss)
    return train_op
