#!/usr/bin/env python3
"""
Batch Normalization Module
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network
    using batch normalization.
    """
    mean = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)

    Z_norm = (Z - mean) / np.sqrt(var + epsilon)
    Z_tilde = gamma * Z_norm + beta
    return Z_tilde
