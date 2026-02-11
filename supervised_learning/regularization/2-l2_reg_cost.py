#!/usr/bin/env python3
"""
L2 Regularization Cost for Keras
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization
    """
    l2_losses = model.losses

    costs = []

    # For each layer, add the base cost + that layer's L2 loss
    for l2_loss in l2_losses:
        costs.append(cost + l2_loss)

    return tf.stack(costs)
