#!/usr/bin/env python3
"""
Dropout Gradient Descent Module
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates weights of a neural network with Dropout regularization
    using gradient descent.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    # Loop through layers in reverse order
    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]

        W = weights['W' + str(layer)]

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Update weights and biases
        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db

        # Calculate gradient for next layer
        if layer > 1:
            # Get dropout mask for previous layer
            D = cache['D' + str(layer - 1)]

            # Backpropagate through tanh activation
            dA = np.matmul(W.T, dZ)

            # Apply dropout mask to gradient
            dA = dA * D

            # Scale gradients to compensate for dropout
            dA = dA / keep_prob
            dZ = dA * (1 - cache['A' + str(layer - 1)] ** 2)
