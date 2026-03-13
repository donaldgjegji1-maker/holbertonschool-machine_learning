#!/usr/bin/env python3
"""Inception Block"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Builds an inception block.

    Args:
        A_prev: output from the previous layer
        filters: tuple/list of (F1, F3R, F3, F5R, F5, FPP)

    Returns:
        Concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # Branch 1: 1x1 convolution
    b1 = K.layers.Conv2D(F1, 1, padding='same', activation='relu')(A_prev)

    # Branch 2: 1x1 reduction -> 3x3 convolution
    b2 = K.layers.Conv2D(F3R, 1, padding='same', activation='relu')(A_prev)
    b2 = K.layers.Conv2D(F3, 3, padding='same', activation='relu')(b2)

    # Branch 3: 1x1 reduction -> 5x5 convolution
    b3 = K.layers.Conv2D(F5R, 1, padding='same', activation='relu')(A_prev)
    b3 = K.layers.Conv2D(F5, 5, padding='same', activation='relu')(b3)

    # Branch 4: 3x3 max pooling -> 1x1 convolution
    b4 = K.layers.MaxPooling2D(3, strides=1, padding='same')(A_prev)
    b4 = K.layers.Conv2D(FPP, 1, padding='same', activation='relu')(b4)

    return K.layers.Concatenate()([b1, b2, b3, b4])
