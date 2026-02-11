#!/usr/bin/env python3
"""
Dropout Gradient Descent
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    # Backpropagate through all layers
    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if layer > 1:
            W = weights['W' + str(layer)]

            dA = np.matmul(W.T, dZ)

            # Apply dropout mask to dA
            D = cache['D' + str(layer - 1)]
            dA = dA * D

            # Scale by keep_prob to maintain expected value
            dA = dA / keep_prob

            # Apply derivative of tanh: (1 - A^2)
            dZ = dA * (1 - np.power(A_prev, 2))

        # Update weights and biases
        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db
