#!/usr/bin/env python3
"""
Build a neural network with Keras using Functional API
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library using Functional API
    """
    inputs = K.Input(shape=(nx,))

    x = inputs

    for i, (layer_size, activation) in enumerate(zip(layers, activations)):
        x = K.layers.Dense(
            units=layer_size,
            activation=activation,
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

        if i < len(layers) - 1:
            x = K.layers.Dropout(rate=1 - keep_prob)(x)
    model = K.Model(inputs=inputs, outputs=x)

    return model
