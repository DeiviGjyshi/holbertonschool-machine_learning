#!/usr/bin/env python3
"""Optimization task"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Learning rate decay upgraded"""
    alpha_new = tf.divide(alpha, tf.pow(1 + decay_rate * 
                          tf.floor_div(global_step, decay_step), 1))
    return alpha_new
