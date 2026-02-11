#!/usr/bin/env python3
"""
L2 Regularization Gradient Descent
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]

        # Calculate gradients with L2 regularization
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Calculate dZ for previous layer before updating weights
        if i > 1:
            dZ = np.matmul(W.T, dZ) * (1 - np.power(A_prev, 2))

        # Update weights and biases
        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db
