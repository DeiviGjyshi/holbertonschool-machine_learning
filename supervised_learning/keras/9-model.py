#!/usr/bin/env python3
"""Keras task"""
import tensorflow.keras as K


def save_model(network, filename):
    """Save model"""
    network.save(filename)
    return None


def load_model(filename):
    """Load model"""
    model = K.models.load_model(filename)
    return model
