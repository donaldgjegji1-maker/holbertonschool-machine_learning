#!/usr/bin/env python3
"""
RMSProp Optimizer Module for TensorFlow
"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow.
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )

    return optimizer
