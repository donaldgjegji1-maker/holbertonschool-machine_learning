#!/usr/bin/env python3
"""
L2 Regularization Layer Creation Module for TensorFlow
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer with L2 regularization.
    """
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # Create L2 regularizer
    regularizer = tf.keras.regularizers.l2(lambtha)

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer,
        name=f'dense_layer_{n}'
    )

    output = layer(prev)
    return output
