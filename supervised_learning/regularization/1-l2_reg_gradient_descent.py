#!/usr/bin/env python3
"""
L2 Regularization Gradient Descent Module
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization
    """
    m = Y.shape[1]

    # Copy weights to use original values during backpropagation
    weights_copy = weights.copy()

    dZ = cache[f'A{L}'] - Y

    # Backpropagate through all layers
    for l in range(L, 0, -1):
        # Get the activation from previous layer
        A_prev = cache[f'A{l - 1}']

        # Calculate gradients with L2 regularization
        dW = (1/m)*np.matmul(dZ, A_prev.T)+(lambtha/m)*weights_copy[f'W{l}']
        db = (1/m)*np.sum(dZ, axis=1, keepdims=True)

        # Update weights and biases in place
        weights[f'W{l}'] -= alpha * dW
        weights[f'b{l}'] -= alpha * db

        # Calculate dZ for previous layer (if not at the first layer)
        if l > 1:
            dA = np.matmul(weights_copy[f'W{l}'].T, dZ)
            # For tanh: derivative is (1 - A^2)
            dZ = dA * (1 - np.square(A_prev))
