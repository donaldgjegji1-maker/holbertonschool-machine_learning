#!/usr/bin/env python3
"""Identity Block"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Builds an identity block.

    Args:
        A_prev: output from the previous layer
        filters: tuple/list of (F11, F3, F12)

    Returns:
        Activated output of the identity block
    """
    F11, F3, F12 = filters
    init = K.initializers.HeNormal(seed=0)

    # 1x1 conv
    x = K.layers.Conv2D(F11, 1, padding='same',
                        kernel_initializer=init)(A_prev)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    # 3x3 conv
    x = K.layers.Conv2D(F3, 3, padding='same',
                        kernel_initializer=init)(x)
    x = K.layers.BatchNormalization(axis=3)(x)
    x = K.layers.Activation('relu')(x)

    # 1x1 conv (restore depth)
    x = K.layers.Conv2D(F12, 1, padding='same',
                        kernel_initializer=init)(x)
    x = K.layers.BatchNormalization(axis=3)(x)

    # Skip connection + final activation
    x = K.layers.Add()([x, A_prev])
    x = K.layers.Activation('relu')(x)

    return x
