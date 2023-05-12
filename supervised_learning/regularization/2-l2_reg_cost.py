#!/usr/bin/env python3
"""L2 regularization"""
import tensorflow as tf


def l2_reg_cost(cost):
    """L2 regularization cost"""
    return cost + tf.losses.get_regularization_losses()
