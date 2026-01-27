#!/usr/bin/env python3
"""
Build a neural network with Keras
"""
import tensorflow as tf


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library
    """
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(
        units=layers[0],
        activation=activations[0],
        kernel_regularizer=tf.keras.regularizers.l2(lambtha),
        input_shape=(nx,)
    ))
    model.add(tf.keras.layers.Dropout(rate=1 - keep_prob))

    for i in range(1, len(layers) - 1):
        model.add(tf.keras.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=tf.keras.regularizers.l2(lambtha)
        ))
        model.add(tf.keras.layers.Dropout(rate=1 - keep_prob))

    model.add(tf.keras.layers.Dense(
        units=layers[-1],
        activation=activations[-1],
        kernel_regularizer=tf.keras.regularizers.l2(lambtha)
    ))

    return model
