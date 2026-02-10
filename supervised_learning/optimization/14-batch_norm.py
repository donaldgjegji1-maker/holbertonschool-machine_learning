#!/usr/bin/env python3
"""
Batch Normalization Layer Module for TensorFlow
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow.
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense_layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )

    Z = dense_layer(prev)

    gamma = tf.Variable(
        initial_value=tf.ones((1, n)),
        trainable=True,
        name='gamma'
    )

    beta = tf.Variable(
        initial_value=tf.zeros((1, n)),
        trainable=True,
        name='beta'
    )

    mean, var = tf.nn.moments(Z, axes=[0], keepdims=True)

    epsilon = 1e-7
    Z_norm = tf.divide(
        tf.subtract(Z, mean),
        tf.sqrt(tf.add(var, epsilon))
    )

    Z_tilde = tf.add(
        tf.multiply(gamma, Z_norm),
        beta
    )

    A = activation(Z_tilde)
    return A
