#!/usr/bin/env python3
"""
L2 Regularization Gradient Descent Module
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases using gradient descent with L2 regularization.
    """
    m = Y.shape[1]

    dZ = cache['A' + str(L)] - Y

    # Loop through layers in reverse order
    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]

        W = weights['W' + str(layer)]

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        dW += (lambtha / m) * W

        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db

        if layer > 1:
            dZ = np.matmul(W.T, dZ) * (1 - cache['A' + str(layer - 1)] ** 2)
