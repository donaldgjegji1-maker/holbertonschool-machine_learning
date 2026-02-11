#!/usr/bin/env python3
"""
L2 Regularization Cost Module for Keras
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.
    """
    l2_costs = tf.zeros(1)

    for layer in model.layers:
        if hasattr(layer, 'losses') and layer.losses:
            # Sum all L2 regularization losses for this layer
            l2_costs = tf.add(l2_costs, tf.reduce_sum(layer.losses))

    total_cost = cost + l2_costs
    return total_cost
