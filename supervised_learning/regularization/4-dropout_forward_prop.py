#!/usr/bin/env python3
"""
Dropout Forward Propagation Module
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L):
        # Linear transformation
        Z = np.matmul(weights[f'W{i}'], cache[f'A{i-1}']) + weights[f'b{i}']

        # Tanh activation
        A = np.tanh(Z)

        # Create dropout mask for current layer
        D = np.random.rand(A.shape[0], A.shape[1])
        D = (D < keep_prob).astype(int)

        # Apply dropout
        A = A * D

        # Scale activations to keep expected value constant
        A = A / keep_prob

        # Store activation and dropout mask
        cache[f'A{i}'] = A
        cache[f'D{i}'] = D

    # Last layer
    Z = np.matmul(weights[f'W{L}'], cache[f'A{L-1}']) + weights[f'b{L}']

    # Softmax activation for last layer
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    # Store output of last layer
    cache[f'A{L}'] = A
    return cache
