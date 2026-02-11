#!/usr/bin/env python3
"""
L2 Regularization Cost Module
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.
    """
    l2_term = 0

    for i in range(1, L + 1):
        weight_key = f'W{i}'
        if weight_key in weights:
            l2_term += np.sum(weights[weight_key] ** 2)

    l2_cost = cost + (lambtha / (2 * m)) * l2_term
    return l2_cost
