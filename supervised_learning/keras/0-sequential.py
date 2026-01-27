#!/usr/bin/env python3
"""
Build a neural network with Keras
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library
    """
    model = K.Sequential()

    model.add(K.layers.Dense(
        units=layers[0],
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha),
        input_shape=(nx,)
    ))
    model.add(K.layers.Dropout(rate=1 - keep_prob))

    for i in range(1, len(layers) - 1):
        model.add(K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        ))
        model.add(K.layers.Dropout(rate=1 - keep_prob))

    model.add(K.layers.Dense(
        units=layers[-1],
        activation=activations[-1],
        kernel_regularizer=K.regularizers.l2(lambtha)
    ))

    return model
