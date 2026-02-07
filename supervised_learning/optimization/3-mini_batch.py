#!/usr/bin/env python3
"""
Mini-Batch Creation Module
"""

import numpy as np

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches to be used for training a neural network using
    mini-batch gradient descent.
    """
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    m = X.shape[0]
    mini_batches = []

    num_complete_batches = m // batch_size

    for k in range(num_complete_batches):
        start_idx = k * batch_size
        end_idx = (k + 1) * batch_size

        X_batch = X_shuffled[start_idx:end_idx]
        Y_batch = Y_shuffled[start_idx:end_idx]

        mini_batches.append((X_batch, Y_batch))

    if m % batch_size != 0:
        start_idx = num_complete_batches * batch_size
        X_batch = X_shuffled[start_idx:]
        Y_batch = Y_shuffled[start_idx:]

        mini_batches.append((X_batch, Y_batch))

    return mini_batches
