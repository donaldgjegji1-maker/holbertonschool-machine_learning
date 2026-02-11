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

    # Start with base cost and add each L2 loss cumulatively
    cumulative_cost = cost
    for l2_loss in l2_losses:
        cumulative_cost = cumulative_cost + l2_loss
        costs.append(cumulative_cost)

    return tf.stack(costs)
