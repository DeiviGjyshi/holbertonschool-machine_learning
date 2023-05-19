#!/usr/bin/env python3
"""Keras task"""
import tensorflow.keras as K


def save_config(network, filename):
    """Save model config"""
    network.model.to_json(filename)
    return None


def load_config(filename):
    """Load model config"""
    K.models.model_from_json(filename)
    return None
