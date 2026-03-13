#!/usr/bin/env python3
"""Dense Block"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block (DenseNet-B bottleneck variant).

    Args:
        X: output from the previous layer
        nb_filters: number of filters in X
        growth_rate: growth rate for the dense block
        layers: number of layers in the dense block

    Returns:
        Concatenated output of each layer and the total number of filters
    """
    init = K.initializers.HeNormal(seed=0)

    for _ in range(layers):
        # Bottleneck: BN -> ReLU -> 1x1 (4 * growth_rate filters)
        x = K.layers.BatchNormalization(axis=3)(X)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(4 * growth_rate, 1, padding='same',
                            kernel_initializer=init)(x)

        # BN -> ReLU -> 3x3 (growth_rate filters)
        x = K.layers.BatchNormalization(axis=3)(x)
        x = K.layers.Activation('relu')(x)
        x = K.layers.Conv2D(growth_rate, 3, padding='same',
                            kernel_initializer=init)(x)

        # Concatenate input with new feature maps
        X = K.layers.Concatenate(axis=3)([X, x])
        nb_filters += growth_rate

    return X, nb_filters
