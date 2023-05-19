#!/usr/bin/env python3
"""Keras task"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """Keras best model"""
    callback = []
    if validation_data is not None and early_stopping:
        callback.append(K.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience= patience))
    if validation_data is not None and learning_rate_decay:
        def lr_sched(epochs):
            return (alpha / (1 + (decay_rate + epochs)))
        callback.append(K.callbacks.LearningRateScheduler(schedule=lr_sched,
                                                          verbose=1))
    if filepath is not None:
        callback.append(K.callbacks.ModelCheckpoint(filepath,
                                                    monitor='val_loss',
                                                    save_best_only=save_best
                                                    ))
    history = network.fit(data, labels,
                          batch_size=batch_size,
                          epochs=epochs, verbose=verbose,
                          validation_data=validation_data,
                          callbacks=callback,
                          shuffle=shuffle)
    return history
