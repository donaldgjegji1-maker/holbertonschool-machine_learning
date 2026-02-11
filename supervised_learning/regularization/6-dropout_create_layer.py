#!/usr/bin/env python3
"""
Create Layer with Dropout
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg'
    )

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )
    output = layer(prev)

    # Dropout rate = 1 - keep_prob
    dropout_layer = tf.keras.layers.Dropout(rate=1 - keep_prob)

    # Apply dropout with training flag
    output = dropout_layer(output, training=training)
    return output
