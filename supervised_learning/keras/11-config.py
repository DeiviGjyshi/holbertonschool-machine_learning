#!/usr/bin/env python3
"""Keras task"""
import tensorflow.keras as K


def save_config(network, filename):
    """Function that saves a model’s configuration in JSON format"""
    model = network.to_json()
    with open(filename, "w") as f:
        f.write(model)
    return None


def load_config(filename):
    """Load model config"""
    with open(filename) as f:
        model = f.read()
    network = K.models.model_from_json(model)
    return network
