#!/usr/bin/env python3
"""Transition Layer"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer (DenseNet-C).

    Args:
        X: output from the previous layer
        nb_filters: number of filters in X
        compression: compression factor (0 < compression <= 1)

    Returns:
        Output of the transition layer and the number of filters
    """
    init = K.initializers.HeNormal(seed=0)
    nb_filters = int(nb_filters * compression)

    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(nb_filters, 1, padding='same',
                        kernel_initializer=init)(X)
    X = K.layers.AveragePooling2D(2, strides=2)(X)

    return X, nb_filters
