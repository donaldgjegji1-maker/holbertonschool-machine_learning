#!/usr/bin/env python3
"""
Create Layer with L2 Regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in TensorFlow
    that includes L2 regularization
    """
    regularizer = tf.keras.regularizers.l2(lambtha)

    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0,
        mode='fan_avg'
    )

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )

    return layer(prev)
