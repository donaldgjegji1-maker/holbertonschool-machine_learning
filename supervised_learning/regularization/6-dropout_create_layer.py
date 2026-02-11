#!/usr/bin/env python3
"""
Dropout Layer Creation Module for TensorFlow
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense_layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )

    # Apply dense layer to previous output
    Z = dense_layer(prev)

    # Create dropout layer
    dropout_layer = tf.keras.layers.Dropout(
        rate=1 - keep_prob,  # Dropout rate = probability of dropping
        seed=4
    )

    A = dropout_layer(Z, training=training)
    return A
