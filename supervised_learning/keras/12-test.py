#!/usr/bin/env python3
"""keras task"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Test the model"""
    results = network.evaluate(data, labels, verbose=verbose)
    return results
