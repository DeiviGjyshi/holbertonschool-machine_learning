#!/usr/bin/env python3
"""Keras task"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """Keras early stopping"""
    callback = []
    if validation_data is not None and early_stopping:
        callback.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=patience))
    history = network.fit(data, labels,
                          batch_size=batch_size,
                          epochs=epochs, verbose=verbose,
                          validation_data=validation_data,
                          callbacks=callback,
                          shuffle=shuffle)
    return history
