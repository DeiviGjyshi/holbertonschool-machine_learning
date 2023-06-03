#!/usr/bin/env python3
"""Deep neural architecture"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Inception network"""
    init = K.initializers.he_normal()
    activation_f = K.activations.relu
    img_input = K.Input(shape=(224, 224, 3))
    c1 = K.layers.Conv2D(filters=64,
                         kernel_size=(7, 7),
                         strides=(2, 2),
                         padding='same',
                         activation=activation_f,
                         kernel_initializer=init)(img_input)
    mp1 = K.layers.MaxPool2D(pool_size=(3, 3),
                             strides=(2, 2),
                             padding='same')(c1)
    c2 = K.layers.Conv2D(filters=64,
                         kernel_size=(1, 1),
                         strides=(1, 1),
                         padding='same',
                         activation=activation_f,
                         kernel_initializer=init)(mp1)
    c3 = K.layers.Conv2D(filters=192,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         padding='same',
                         activation=activation_f,
                         kernel_initializer=init)(c2)
    mp2 = K.layers.MaxPool2D(pool_size=(3, 3),
                             strides=(2, 2),
                             padding='same')(c3)
    I1 = inception_block(mp2, [64, 96, 128, 16, 32, 32])
    I2 = inception_block(I1, [128, 128, 192, 32, 96, 64])
    mp3 = K.layers.MaxPool2D(pool_size=(3, 3),
                             strides=(2, 2),
                             padding='same')(I2)
    I3 = inception_block(mp3, [192, 96, 208, 16, 48, 64])
    I4 = inception_block(I3, [160, 112, 224, 24, 64, 64])
    I5 = inception_block(I4, [128, 128, 256, 24, 64, 64])
    I6 = inception_block(I5, [112, 144, 288, 32, 64, 64])
    I7 = inception_block(I6, [256, 160, 320, 32, 128, 128])
    mp4 = K.layers.MaxPool2D(pool_size=(3, 3),
                             strides=(2, 2),
                             padding='same')(I7)
    I8 = inception_block(mp4, [256, 160, 320, 32, 128, 128])
    I9 = inception_block(I8, [384, 192, 384, 48, 128, 128])
    AP1 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                    strides=(1, 1),
                                    padding='valid')(I9)
    Dropout1 = K.layers.Dropout(rate=0.4)(AP1)
    output = K.layers.Dense(1000,
                            activation='softmax',
                            kernel_initializer=init)(Dropout1)
    model = K.Model(inputs=img_input, outputs=output)
    return model
