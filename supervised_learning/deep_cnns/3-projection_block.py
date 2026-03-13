#!/usr/bin/env python3
"""Projection Block"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block.

    Args:
        A_prev: output from the previous layer
        filters: tuple/list of (F11, F3, F12)
        s: stride of the first convolution in main path and shortcut

    Returns:
        Activated output of the projection block
    """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    # Main path
    x = K.layers.Conv2D(F11, 1, strides=s,
                        kernel_initializer=init)(A_prev)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.ReLU()(x)

    x = K.layers.Conv2D(F3, 3, padding='same',
                        kernel_initializer=init)(x)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.ReLU()(x)

    x = K.layers.Conv2D(F12, 1,
                        kernel_initializer=init)(x)
    x = K.layers.BatchNormalization(axis=3)(x)

    # Shortcut connection
    shortcut = K.layers.Conv2D(F12, 1, strides=s,
                               kernel_initializer=init)(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Add + final activation
    x = K.layers.Add()([x, shortcut])
    x = K.layers.ReLU()(x)

    return x
